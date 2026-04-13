/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2026
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

#ifndef iterReduceIncompressibleFluid_H
#define iterReduceIncompressibleFluid_H

#include "incompressibleFluidIterReduce.H"
#include "IterReduceCoupler.H"

namespace Foam
{
namespace solvers
{

class iterReduceIncompressibleFluid
:
    public incompressibleFluidIterReduce
{
    //- OpenFOAM<->GNN coupling helper
    mutable IterReduceCoupler iterReduceCoupler_;

public:
    TypeName("iterReduceIncompressibleFluid");

    explicit iterReduceIncompressibleFluid(fvMesh& mesh);
    virtual ~iterReduceIncompressibleFluid();

    //- Inject GNN fields once per time-step after prePredictor()
    bool injectPredictorFields();

private:
    //- Refresh flux state so the predictor step sees fields consistent with injected U/p
    void refreshFluxStateAfterInjection();
};

} // End namespace solvers
} // End namespace Foam

#endif

// ************************************************************************* //
