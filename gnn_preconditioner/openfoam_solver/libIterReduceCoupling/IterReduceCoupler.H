/*---------------------------------------------------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           |
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

#ifndef IterReduceCoupler_H
#define IterReduceCoupler_H

#include "fvMesh.H"
#include "volFields.H"
#include <memory>

namespace Foam
{

class IterReduceCoupler
{
    const fvMesh& mesh_;

    bool enabled_;
    bool injectEveryTimeStep_;
    bool applyPressureGuess_;
    bool preserveUz_;
    bool abortOnError_;
    bool logGuessDiagnostics_;
    bool dumpGuessFields_;
    scalar startInjectTime_;
    scalar macroDt_;
    scalar guessBlendAlpha_;
    scalar gnnTrainingDt_;
    scalar dtScaledAlphaMax_;
    scalar timeTol_;
    scalar nextInjectionTime_;
    label lastInjectedTimeIndex_;
    label dumpGuessFieldLimit_;
    word device_;
    word predictorMode_;
    word blendMode_;
    word pressureGaugeMode_;
    fileName modelPath_;
    fileName contextDir_;

    mutable bool runtimeLoaded_;
    mutable bool runtimeLoadAttempted_;
    mutable label numInjections_;
    mutable label numDumpedFieldPairs_;
    mutable scalar totalInjectWallTime_;
    mutable scalar totalForwardWallTime_;

public:
    class RuntimeData;

private:
    mutable std::unique_ptr<RuntimeData> runtime_;

    void readControls();
    void handleError(const string& message) const;
    void ensureRuntimeLoaded(const volVectorField& U) const;
    scalar inferInletUx(const volVectorField& U) const;
    void advanceNextInjectionTime(const scalar currentTime);
    void logTimingSummary() const;
    void dumpInjectionFields
    (
        const volVectorField& U,
        const volScalarField& p,
        const vectorField& Uprev,
        const scalarField& pprev,
        const scalar currentTime,
        const label timeIndex
    ) const;

public:
    explicit IterReduceCoupler(const fvMesh& mesh);
    ~IterReduceCoupler();

    bool enabled() const
    {
        return enabled_;
    }

    bool shouldInject(const scalar currentTime, const label timeIndex) const;

    bool inject
    (
        volVectorField& U,
        volScalarField& p,
        const scalar currentTime,
        const label timeIndex
    );
};

} // End namespace Foam

#endif

// ************************************************************************* //
