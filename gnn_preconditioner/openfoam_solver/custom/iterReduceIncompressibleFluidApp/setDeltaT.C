/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2022-2023 OpenFOAM Foundation
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

#include "setDeltaT.H"

void Foam::setDeltaT(Time& runTime, const solver& solver)
{
    if
    (
        runTime.timeIndex() == 0
     && runTime.controlDict().lookupOrDefault("adjustTimeStep", false)
     && solver.transient()
    )
    {
        const scalar deltaT =
            min(solver.maxDeltaT(), runTime.functionObjects().maxDeltaT());

        if (deltaT < rootVGreat)
        {
            runTime.setDeltaT(min(runTime.deltaTValue(), deltaT));
        }
    }
}


void Foam::adjustDeltaT(Time& runTime, const solver& solver)
{
    // Update the time-step limited by the solver maxDeltaT
    if
    (
        runTime.controlDict().lookupOrDefault("adjustTimeStep", false)
     && solver.transient()
    )
    {
        const scalar deltaT =
            min(solver.maxDeltaT(), runTime.functionObjects().maxDeltaT());

        if (deltaT < rootVGreat)
        {
            runTime.setDeltaT
            (
                min(solver::deltaTFactor*runTime.deltaTValue(), deltaT)
            );
            Info<< "deltaT = " <<  runTime.deltaTValue() << endl;
        }
    }
}

// ************************************************************************* //
