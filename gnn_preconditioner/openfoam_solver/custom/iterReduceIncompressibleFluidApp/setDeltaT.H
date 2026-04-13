/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2022 OpenFOAM Foundation
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

#ifndef setDeltaT_H
#define setDeltaT_H

#include "solver.H"

namespace Foam
{

//- Set the initial time-step according to the solver maxDeltaT
void setDeltaT(Time& runTime, const solver& solver);

//- Adjust the time-step according to the solver maxDeltaT
void adjustDeltaT(Time& runTime, const solver& solver);

} // End namespace Foam

#endif

// ************************************************************************* //
