/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2026
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

#include "iterReduceIncompressibleFluid.H"
#include "addToRunTimeSelectionTable.H"
#include "linear.H"
#include "fvc.H"

namespace Foam
{
namespace solvers
{
    defineTypeNameAndDebug(iterReduceIncompressibleFluid, 0);
    addToRunTimeSelectionTable(solver, iterReduceIncompressibleFluid, fvMesh);
}
}


Foam::solvers::iterReduceIncompressibleFluid::iterReduceIncompressibleFluid(fvMesh& mesh)
:
    incompressibleFluidIterReduce(mesh),
    iterReduceCoupler_(mesh)
{
    injectInitialPredictorFields();
}


Foam::solvers::iterReduceIncompressibleFluid::~iterReduceIncompressibleFluid()
{}


void Foam::solvers::iterReduceIncompressibleFluid::refreshFluxStateAfterInjection
(
    const bool refreshPhi,
    const bool refreshUf
)
{
    if (!refreshPhi && !refreshUf)
    {
        return;
    }

    const scalarField phiPrev(phi_.primitiveField());

    if (refreshPhi)
    {
        phi_ = linearInterpolate(U_) & mesh_.Sf();

        MRF.makeRelative(phi_);
        fvc::makeRelative(phi_, U_);
    }

    if (refreshUf && Uf.valid())
    {
        Uf() = fvc::interpolate(U_);
    }

    scalar sumDPhi2 = 0;
    scalar sumPhi2 = 0;
    scalar maxDPhi = 0;
    const scalarField& phiNow = phi_.primitiveField();
    forAll(phiNow, facei)
    {
        const scalar dPhi = phiNow[facei] - phiPrev[facei];
        sumDPhi2 += sqr(dPhi);
        sumPhi2 += sqr(phiPrev[facei]);
        maxDPhi = max(maxDPhi, mag(dPhi));
    }

    const scalar dPhiNorm = sqrt(sumDPhi2);
    const scalar phiNorm = sqrt(sumPhi2);
    const scalar dPhiRel = (phiNorm > SMALL) ? dPhiNorm/phiNorm : 0.0;

    Info<< "iterReduceCoupling: refreshed flux state"
        << " refreshPhi=" << refreshPhi
        << " refreshUf=" << refreshUf
        << " ||dphi||2=" << dPhiNorm
        << " dphi_relL2=" << dPhiRel
        << " dphi_max=" << maxDPhi
        << nl;
}


bool Foam::solvers::iterReduceIncompressibleFluid::injectPredictorFields()
{
    const IterReduceCoupler::InjectionOutcome outcome =
        iterReduceCoupler_.inject(U_, p_, runTime.value(), runTime.timeIndex());

    if (outcome.refreshPhi || outcome.refreshUf)
    {
        refreshFluxStateAfterInjection(outcome.refreshPhi, outcome.refreshUf);
    }

    return outcome.scheduled;
}


bool Foam::solvers::iterReduceIncompressibleFluid::injectInitialPredictorFields()
{
    if (!iterReduceCoupler_.injectAtInitialTime())
    {
        return false;
    }

    Info<< "iterReduceCoupling: initial injection check"
        << " time=" << runTime.value()
        << " timeIndex=" << runTime.timeIndex()
        << nl;

    const IterReduceCoupler::InjectionOutcome outcome =
        iterReduceCoupler_.inject(U_, p_, runTime.value(), runTime.timeIndex());

    if (outcome.refreshPhi || outcome.refreshUf)
    {
        refreshFluxStateAfterInjection(outcome.refreshPhi, outcome.refreshUf);
    }

    return outcome.scheduled;
}


// ************************************************************************* //
