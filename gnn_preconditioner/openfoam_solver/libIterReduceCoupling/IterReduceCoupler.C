/*---------------------------------------------------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           |
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

#include "IterReduceCoupler.H"

#include "IOdictionary.H"
#include "Time.H"
#include "error.H"
#include "fvPatchField.H"

#ifdef TypeName
#pragma push_macro("TypeName")
#undef TypeName
#define FOAM_RESTORE_TYPENAME_MACRO
#endif
#include <torch/script.h>
#include <torch/serialize.h>
#include <torch/cuda.h>
#ifdef FOAM_RESTORE_TYPENAME_MACRO
#pragma pop_macro("TypeName")
#undef FOAM_RESTORE_TYPENAME_MACRO
#endif

#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace Foam
{

class IterReduceCoupler::RuntimeData
{
public:
    torch::jit::script::Module module;
    torch::Device device{torch::kCPU};
    label nCells{0};
};

namespace
{
using steady_clock = std::chrono::steady_clock;

static scalar elapsedSeconds
(
    const steady_clock::time_point& start,
    const steady_clock::time_point& end
)
{
    return scalar(std::chrono::duration<double>(end - start).count());
}


static void synchronizeIfCuda(const torch::Device& device)
{
    if (device.is_cuda())
    {
        torch::cuda::synchronize(device.index());
    }
}


static bool pathIsFile(const fileName& p)
{
    std::error_code ec;
    return std::filesystem::is_regular_file(std::filesystem::path(p.c_str()), ec);
}


static bool pathIsDir(const fileName& p)
{
    std::error_code ec;
    return std::filesystem::is_directory(std::filesystem::path(p.c_str()), ec);
}


static fileName injectionDumpDir(const fvMesh& mesh, const label timeIndex)
{
    return
        mesh.time().path()
      / "postProcessing"
      / "iterReduceInjectionDumps"
      / ("timeIndex_" + Foam::name(timeIndex));
}


static torch::Device parseDevice(const word& deviceWord)
{
    const std::string s(deviceWord.c_str());

    if (s == "cpu")
    {
        return torch::Device(torch::kCPU);
    }

    if (s.rfind("cuda", 0) == 0)
    {
        if (!torch::cuda::is_available())
        {
            throw std::runtime_error("CUDA device requested but torch::cuda::is_available() is false");
        }

        const auto colonPos = s.find(':');
        if (colonPos == std::string::npos)
        {
            return torch::Device(torch::kCUDA);
        }

        const int index = std::stoi(s.substr(colonPos + 1));
        return torch::Device(torch::kCUDA, index);
    }

    throw std::runtime_error("Unsupported device in iterReduceCouplingDict: " + s);
}


} // End anonymous namespace


IterReduceCoupler::IterReduceCoupler(const fvMesh& mesh)
:
    mesh_(mesh),
    enabled_(true),
    injectAtInitialTime_(false),
    injectEveryTimeStep_(false),
    applyVelocityGuess_(true),
    applyPressureGuess_(true),
    correctBoundariesAfterInjection_(true),
    refreshPhiAfterInjection_(false),
    refreshUfAfterInjection_(false),
    skipSmallUpdates_(true),
    preserveUz_(true),
    abortOnError_(true),
    logGuessDiagnostics_(true),
    dumpGuessFields_(false),
    startInjectTime_(0),
    macroDt_(0.01),
    guessBlendAlpha_(1.0),
    gnnTrainingDt_(0.01),
    dtScaledAlphaMax_(1.0),
    minRelativeVelocityUpdate_(1e-7),
    minRelativePressureUpdate_(1e-7),
    timeTol_(1e-9),
    nextInjectionTime_(0),
    lastInjectedTimeIndex_(-1),
    dumpGuessFieldLimit_(0),
    device_("cpu"),
    predictorMode_("model"),
    blendMode_("manual"),
    pressureGaugeMode_("mean"),
    modelPath_("constant/iterreduce/model_inference.ts"),
    contextDir_("constant/iterreduce/context"),
    runtimeLoaded_(false),
    runtimeLoadAttempted_(false),
    numInjections_(0),
    numDumpedFieldPairs_(0),
    totalInjectWallTime_(0),
    totalForwardWallTime_(0),
    runtime_(nullptr)
{
    readControls();
    nextInjectionTime_ = startInjectTime_;
}


IterReduceCoupler::~IterReduceCoupler()
{
    logTimingSummary();
}


void IterReduceCoupler::readControls()
{
    IOobject io
    (
        "iterReduceCouplingDict",
        mesh_.time().system(),
        mesh_,
        IOobject::READ_IF_PRESENT,
        IOobject::NO_WRITE,
        false
    );

    if (!io.headerOk())
    {
        WarningInFunction
            << "system/iterReduceCouplingDict not found; using runtime defaults" << nl;
        return;
    }

    IOdictionary dict(io);
    enabled_ = dict.lookupOrDefault<bool>("enabled", enabled_);
    injectAtInitialTime_ = dict.lookupOrDefault<bool>("injectAtInitialTime", injectAtInitialTime_);
    startInjectTime_ = dict.lookupOrDefault<scalar>("startInjectTime", startInjectTime_);
    injectEveryTimeStep_ = dict.lookupOrDefault<bool>("injectEveryTimeStep", injectEveryTimeStep_);
    macroDt_ = dict.lookupOrDefault<scalar>("macroDt", macroDt_);
    modelPath_ = dict.lookupOrDefault<fileName>("modelPath", modelPath_);
    contextDir_ = dict.lookupOrDefault<fileName>("contextDir", contextDir_);
    applyVelocityGuess_ = dict.lookupOrDefault<bool>("applyVelocityGuess", applyVelocityGuess_);
    applyPressureGuess_ = dict.lookupOrDefault<bool>("applyPressureGuess", applyPressureGuess_);
    correctBoundariesAfterInjection_ = dict.lookupOrDefault<bool>
    (
        "correctBoundariesAfterInjection",
        correctBoundariesAfterInjection_
    );
    refreshPhiAfterInjection_ = dict.lookupOrDefault<bool>
    (
        "refreshPhiAfterInjection",
        refreshPhiAfterInjection_
    );
    refreshUfAfterInjection_ = dict.lookupOrDefault<bool>
    (
        "refreshUfAfterInjection",
        refreshUfAfterInjection_
    );
    skipSmallUpdates_ = dict.lookupOrDefault<bool>("skipSmallUpdates", skipSmallUpdates_);
    preserveUz_ = dict.lookupOrDefault<bool>("preserveUz", preserveUz_);
    guessBlendAlpha_ = dict.lookupOrDefault<scalar>("guessBlendAlpha", guessBlendAlpha_);
    minRelativeVelocityUpdate_ = dict.lookupOrDefault<scalar>
    (
        "minRelativeVelocityUpdate",
        minRelativeVelocityUpdate_
    );
    minRelativePressureUpdate_ = dict.lookupOrDefault<scalar>
    (
        "minRelativePressureUpdate",
        minRelativePressureUpdate_
    );
    pressureGaugeMode_ = dict.lookupOrDefault<word>("pressureGaugeMode", pressureGaugeMode_);
    abortOnError_ = dict.lookupOrDefault<bool>("abortOnError", abortOnError_);
    logGuessDiagnostics_ = dict.lookupOrDefault<bool>("logGuessDiagnostics", logGuessDiagnostics_);
    dumpGuessFields_ = dict.lookupOrDefault<bool>("dumpGuessFields", dumpGuessFields_);
    dumpGuessFieldLimit_ = dict.lookupOrDefault<label>("dumpGuessFieldLimit", dumpGuessFieldLimit_);
    device_ = dict.lookupOrDefault<word>("device", device_);
    predictorMode_ = dict.lookupOrDefault<word>("predictorMode", predictorMode_);
    blendMode_ = dict.lookupOrDefault<word>("blendMode", blendMode_);
    gnnTrainingDt_ = dict.lookupOrDefault<scalar>("gnnTrainingDt", gnnTrainingDt_);
    dtScaledAlphaMax_ = dict.lookupOrDefault<scalar>("dtScaledAlphaMax", dtScaledAlphaMax_);
    guessBlendAlpha_ = min(max(guessBlendAlpha_, scalar(0)), scalar(1));
    dtScaledAlphaMax_ = min(max(dtScaledAlphaMax_, scalar(0)), scalar(1));
    minRelativeVelocityUpdate_ = max(minRelativeVelocityUpdate_, scalar(0));
    minRelativePressureUpdate_ = max(minRelativePressureUpdate_, scalar(0));

    if (blendMode_ != "manual" && blendMode_ != "dtScaled")
    {
        WarningInFunction
            << "Unknown blendMode '" << blendMode_
            << "'; falling back to 'manual'" << nl;
        blendMode_ = "manual";
    }

    if (blendMode_ == "dtScaled" && gnnTrainingDt_ <= SMALL)
    {
        WarningInFunction
            << "blendMode=dtScaled requires gnnTrainingDt > 0; got "
            << gnnTrainingDt_ << "; falling back to 'manual'" << nl;
        blendMode_ = "manual";
    }
}


void IterReduceCoupler::handleError(const string& message) const
{
    if (abortOnError_)
    {
        FatalErrorInFunction << message << exit(FatalError);
    }
    else
    {
        WarningInFunction << message << nl;
    }
}


scalar IterReduceCoupler::inferInletUx(const volVectorField& U) const
{
    const fvBoundaryMesh& patches = mesh_.boundary();
    forAll(patches, patchi)
    {
        std::string patchName(patches[patchi].name().c_str());
        std::transform
        (
            patchName.begin(),
            patchName.end(),
            patchName.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); }
        );

        if (patchName.find("inlet") != std::string::npos)
        {
            const fvPatchVectorField& Up = U.boundaryField()[patchi];
            if (Up.size() == 0)
            {
                continue;
            }

            scalar sumUx = 0;
            forAll(Up, facei)
            {
                sumUx += Up[facei].x();
            }
            return sumUx/scalar(Up.size());
        }
    }

    return 0;
}


void IterReduceCoupler::ensureRuntimeLoaded(const volVectorField& U) const
{
    if (runtimeLoaded_)
    {
        return;
    }
    if (runtimeLoadAttempted_)
    {
        handleError("iterreduce runtime load previously failed");
        return;
    }

    runtimeLoadAttempted_ = true;
    const steady_clock::time_point t0 = steady_clock::now();

    try
    {
        const fileName resolvedModel = mesh_.time().path()/modelPath_;
        const fileName resolvedContext = mesh_.time().path()/contextDir_;
        if (!pathIsFile(resolvedModel))
        {
            throw std::runtime_error("TorchScript model not found: " + std::string(resolvedModel.c_str()));
        }
        if (!pathIsDir(resolvedContext))
        {
            throw std::runtime_error("Context directory not found: " + std::string(resolvedContext.c_str()));
        }

        runtime_.reset(new RuntimeData());
        runtime_->device = parseDevice(device_);
        runtime_->module = torch::jit::load(std::string(resolvedModel.c_str()), runtime_->device);
        runtime_->module.eval();
        runtime_->nCells = mesh_.nCells();

        runtimeLoaded_ = true;
        synchronizeIfCuda(runtime_->device);
        const scalar loadWall = elapsedSeconds(t0, steady_clock::now());
        Info<< "iterReduceCoupling: runtime loaded"
            << " device=" << device_
            << " nCells=" << runtime_->nCells
            << " loadWall=" << loadWall << " s"
            << nl;
    }
    catch (const std::exception& e)
    {
        runtime_.reset();
        handleError("Failed to load iterreduce runtime: " + string(e.what()));
    }
}


void IterReduceCoupler::advanceNextInjectionTime(const scalar currentTime)
{
    if (injectEveryTimeStep_ || macroDt_ <= SMALL)
    {
        return;
    }

    if (nextInjectionTime_ < startInjectTime_)
    {
        nextInjectionTime_ = startInjectTime_;
    }

    while (nextInjectionTime_ <= currentTime + timeTol_)
    {
        nextInjectionTime_ += macroDt_;
    }
}


void IterReduceCoupler::logTimingSummary() const
{
    if (numInjections_ <= 0)
    {
        return;
    }

    Info<< "iterReduceCoupling: injections=" << numInjections_
        << " totalInjectWall=" << totalInjectWallTime_
        << " totalForwardWall=" << totalForwardWallTime_
        << nl;
}


void IterReduceCoupler::dumpInjectionFields
(
    const volVectorField& U,
    const volScalarField& p,
    const vectorField& Uprev,
    const scalarField& pprev,
    const scalar currentTime,
    const label timeIndex
) const
{
    if (!dumpGuessFields_)
    {
        return;
    }
    if (dumpGuessFieldLimit_ > 0 && numDumpedFieldPairs_ >= dumpGuessFieldLimit_)
    {
        return;
    }

    const fileName outDir = injectionDumpDir(mesh_, timeIndex);
    std::filesystem::create_directories(std::filesystem::path(outDir.c_str()));

    auto writeCsv = [&](const fileName& path, const vectorField& vel, const scalarField& pres)
    {
        std::ofstream os(path.c_str());
        os << "cellI,x,y,ux,uy,p,time\n";
        const vectorField& centers = mesh_.C();
        forAll(vel, celli)
        {
            os << celli
               << ',' << centers[celli].x()
               << ',' << centers[celli].y()
               << ',' << vel[celli].x()
               << ',' << vel[celli].y()
               << ',' << pres[celli]
               << ',' << currentTime
               << '\n';
        }
    };

    writeCsv(outDir/("state_before_ti" + Foam::name(timeIndex) + ".csv"), Uprev, pprev);
    writeCsv(outDir/("state_after_ti" + Foam::name(timeIndex) + ".csv"), U.primitiveField(), p.primitiveField());
    ++numDumpedFieldPairs_;
}


bool IterReduceCoupler::shouldInject(const scalar currentTime, const label timeIndex) const
{
    if (!enabled_)
    {
        return false;
    }
    if (timeIndex == lastInjectedTimeIndex_)
    {
        return false;
    }
    if (currentTime + timeTol_ < startInjectTime_)
    {
        return false;
    }
    if (injectEveryTimeStep_)
    {
        return true;
    }
    return currentTime + timeTol_ >= nextInjectionTime_;
}


IterReduceCoupler::InjectionOutcome IterReduceCoupler::inject
(
    volVectorField& U,
    volScalarField& p,
    const scalar currentTime,
    const label timeIndex
)
{
    InjectionOutcome outcome;

    if (!shouldInject(currentTime, timeIndex))
    {
        return outcome;
    }

    outcome.scheduled = true;
    const steady_clock::time_point tStart = steady_clock::now();

    if (predictorMode_ == "identityNoop")
    {
        if (logGuessDiagnostics_)
        {
            Info<< "iterReduceCoupling: identityNoop"
                << " time=" << currentTime
                << " timeIndex=" << timeIndex
                << nl;
        }

        lastInjectedTimeIndex_ = timeIndex;
        ++numInjections_;
        advanceNextInjectionTime(currentTime);
        totalInjectWallTime_ += elapsedSeconds(tStart, steady_clock::now());
        return outcome;
    }

    if (pressureGaugeMode_ != "mean")
    {
        handleError("Unsupported pressureGaugeMode; expected 'mean'");
        return outcome;
    }

    const vectorField Uprev(U.primitiveField());
    const scalarField pprev(p.primitiveField());
    const label nCells = Uprev.size();
    const scalar meanP = (nCells > 0 ? gSum(pprev)/scalar(nCells) : 0);
    const scalar inletUx = inferInletUx(U);
    const bool identityMode = (predictorMode_ == "identity");

    torch::Tensor predUxyCpu;
    torch::Tensor predPgCpu;

    if (!identityMode)
    {
        ensureRuntimeLoaded(U);
        if (!runtimeLoaded_)
        {
            return outcome;
        }

        try
        {
            const auto tensorOpts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
            torch::Tensor uxyCpu = torch::empty({nCells, 2}, tensorOpts);
            torch::Tensor pgCpu = torch::empty({nCells, 1}, tensorOpts);
            float* uxyPtr = uxyCpu.data_ptr<float>();
            float* pgPtr = pgCpu.data_ptr<float>();
            forAll(Uprev, celli)
            {
                uxyPtr[2*celli + 0] = float(Uprev[celli].x());
                uxyPtr[2*celli + 1] = float(Uprev[celli].y());
                pgPtr[celli] = float(pprev[celli] - meanP);
            }

            if (predictorMode_ != "model")
            {
                throw std::runtime_error("Unsupported predictorMode: " + std::string(predictorMode_.c_str()));
            }

            torch::Tensor uxyDev = uxyCpu.to(runtime_->device);
            torch::Tensor pgDev = pgCpu.to(runtime_->device);
            torch::Tensor inletDev = torch::tensor({float(inletUx)}, torch::TensorOptions().dtype(torch::kFloat32).device(runtime_->device));
            std::vector<torch::jit::IValue> inputs;
            inputs.emplace_back(uxyDev);
            inputs.emplace_back(pgDev);
            inputs.emplace_back(inletDev);
            const steady_clock::time_point tFwd0 = steady_clock::now();
            torch::NoGradGuard noGrad;
            const auto out = runtime_->module.forward(inputs).toTuple();
            synchronizeIfCuda(runtime_->device);
            totalForwardWallTime_ += elapsedSeconds(tFwd0, steady_clock::now());
            predUxyCpu = out->elements()[0].toTensor().to(torch::kCPU).to(torch::kFloat32).contiguous();
            predPgCpu = out->elements()[1].toTensor().to(torch::kCPU).to(torch::kFloat32).contiguous();
        }
        catch (const std::exception& e)
        {
            handleError("iterreduce forward/inference failure: " + string(e.what()));
            return outcome;
        }

        if (label(predUxyCpu.size(0)) != nCells || label(predUxyCpu.size(1)) != 2)
        {
            handleError("TorchScript u_xy_guess shape does not match mesh cell count");
            return outcome;
        }
        if (label(predPgCpu.size(0)) != nCells || label(predPgCpu.size(1)) != 1)
        {
            handleError("TorchScript p_gauge_guess shape does not match mesh cell count");
            return outcome;
        }
    }

    const float* predUPtr = identityMode ? nullptr : predUxyCpu.data_ptr<float>();
    const float* predPPtr = identityMode ? nullptr : predPgCpu.data_ptr<float>();
    vectorField& Ufld = U.primitiveFieldRef();
    scalarField& pfld = p.primitiveFieldRef();

    const scalar dtOF = mesh_.time().deltaTValue();
    scalar alphaEff = guessBlendAlpha_;
    if (blendMode_ == "dtScaled")
    {
        const scalar ratio = dtOF / gnnTrainingDt_;
        alphaEff = min(max(ratio, scalar(0)), dtScaledAlphaMax_);
    }

    scalar sumDU2 = 0;
    scalar sumU2 = 0;
    scalar sumDP2 = 0;
    scalar sumP2 = 0;
    forAll(Uprev, celli)
    {
        const scalar guessUx = identityMode ? Uprev[celli].x() : scalar(predUPtr[2*celli + 0]);
        const scalar guessUy = identityMode ? Uprev[celli].y() : scalar(predUPtr[2*celli + 1]);
        const scalar outOfPlane = preserveUz_ ? Uprev[celli].z() : 0.0;
        const vector guessU(guessUx, guessUy, outOfPlane);
        const vector blendedU = (1.0 - alphaEff)*Uprev[celli] + alphaEff*guessU;
        const scalar guessP = identityMode ? pprev[celli] : scalar(predPPtr[celli]) + meanP;
        const scalar blendedP = (1.0 - alphaEff)*pprev[celli] + alphaEff*guessP;

        if (applyVelocityGuess_)
        {
            sumDU2 += magSqr(blendedU - Uprev[celli]);
        }
        sumU2 += magSqr(Uprev[celli]);
        if (applyPressureGuess_)
        {
            sumDP2 += sqr(blendedP - pprev[celli]);
            sumP2 += sqr(pprev[celli]);
        }
    }

    const scalar dUNorm = sqrt(sumDU2);
    const scalar uNorm = sqrt(sumU2);
    const scalar dURel = (uNorm > SMALL) ? dUNorm/uNorm : dUNorm;
    const scalar dPNorm = sqrt(sumDP2);
    const scalar pNorm = sqrt(sumP2);
    const scalar dPRel = (pNorm > SMALL) ? dPNorm/pNorm : dPNorm;

    const bool writeVelocity =
        applyVelocityGuess_
     && (
            !skipSmallUpdates_
         || (
                dUNorm > SMALL
             && (uNorm <= SMALL || dURel > minRelativeVelocityUpdate_)
            )
        );
    const bool writePressure =
        applyPressureGuess_
     && (
            !skipSmallUpdates_
         || (
                dPNorm > SMALL
             && (pNorm <= SMALL || dPRel > minRelativePressureUpdate_)
            )
        );

    if (writeVelocity || writePressure)
    {
        forAll(Ufld, celli)
        {
            const scalar guessUx = identityMode ? Uprev[celli].x() : scalar(predUPtr[2*celli + 0]);
            const scalar guessUy = identityMode ? Uprev[celli].y() : scalar(predUPtr[2*celli + 1]);
            const scalar outOfPlane = preserveUz_ ? Uprev[celli].z() : 0.0;
            const vector guessU(guessUx, guessUy, outOfPlane);
            const vector blendedU = (1.0 - alphaEff)*Uprev[celli] + alphaEff*guessU;
            const scalar guessP = identityMode ? pprev[celli] : scalar(predPPtr[celli]) + meanP;
            const scalar blendedP = (1.0 - alphaEff)*pprev[celli] + alphaEff*guessP;

            if (writeVelocity)
            {
                Ufld[celli] = blendedU;
            }
            if (writePressure)
            {
                pfld[celli] = blendedP;
            }
        }

        if (correctBoundariesAfterInjection_)
        {
            if (writeVelocity)
            {
                U.correctBoundaryConditions();
            }
            if (writePressure)
            {
                p.correctBoundaryConditions();
            }
        }
    }

    if (logGuessDiagnostics_)
    {
        Info<< "iterReduceCoupling: injected guess"
            << " time=" << currentTime
            << " timeIndex=" << timeIndex
            << " predictorMode=" << predictorMode_
            << " blendMode=" << blendMode_
            << " dtOF=" << dtOF
            << " dtGNN=" << gnnTrainingDt_
            << " alphaEff=" << alphaEff
            << " inletUx=" << inletUx
            << " applyU=" << applyVelocityGuess_
            << " applyP=" << applyPressureGuess_
            << " wroteU=" << writeVelocity
            << " wroteP=" << writePressure
            << " correctBC=" << correctBoundariesAfterInjection_
            << " refreshPhi=" << (writeVelocity && refreshPhiAfterInjection_)
            << " refreshUf=" << (writeVelocity && refreshUfAfterInjection_)
            << " ||dUxy||2=" << dUNorm
            << " dUxy_relL2=" << dURel;
        if (applyPressureGuess_)
        {
            Info<< " ||dp||2=" << dPNorm
                << " dp_relL2=" << dPRel
                << " meanP_current=" << meanP;
        }
        Info<< nl;
    }

    dumpInjectionFields(U, p, Uprev, pprev, currentTime, timeIndex);

    outcome.wroteVelocity = writeVelocity;
    outcome.wrotePressure = writePressure;
    outcome.refreshPhi = writeVelocity && refreshPhiAfterInjection_;
    outcome.refreshUf = writeVelocity && refreshUfAfterInjection_;

    lastInjectedTimeIndex_ = timeIndex;
    ++numInjections_;
    advanceNextInjectionTime(currentTime);
    totalInjectWallTime_ += elapsedSeconds(tStart, steady_clock::now());
    return outcome;
}

} // End namespace Foam

// ************************************************************************* //
