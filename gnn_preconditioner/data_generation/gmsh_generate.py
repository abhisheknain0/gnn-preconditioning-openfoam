#!/usr/bin/env python3
"""Gmsh geometry generation for the canonical empty-based BFS stack."""

from __future__ import annotations

from pathlib import Path
import subprocess

from .common import MeshProfile


XMIN = 0.0
XMAX = 0.4
YMIN = 0.0
THICKNESS = 0.001


def generate_gmsh_geo(
    output_path: Path,
    ymid: float,
    xmid: float,
    ymax: float,
    profile: MeshProfile,
) -> Path:
    output_path = Path(output_path)
    geo_content = f"""// Canonical empty-based BFS geometry for gnn_preconditioner
// Geometry units are meters.

lc_coarse = {profile.mesh_size_coarse};
lc_fine = {profile.mesh_size_fine};

Point(1) = {{{XMIN}, {ymid}, 0, lc_coarse}};
Point(2) = {{{xmid}, {ymid}, 0, lc_fine}};
Point(3) = {{{xmid}, {ymax}, 0, lc_fine}};
Point(4) = {{{XMIN}, {ymax}, 0, lc_coarse}};
Point(5) = {{{xmid}, {YMIN}, 0, lc_fine}};
Point(6) = {{{XMAX}, {YMIN}, 0, lc_fine}};
Point(7) = {{{XMAX}, {ymid}, 0, lc_fine}};
Point(8) = {{{XMAX}, {ymax}, 0, lc_fine}};

Line(1) = {{1, 2}};
Line(2) = {{2, 3}};
Line(3) = {{3, 4}};
Line(4) = {{4, 1}};
Line(5) = {{5, 6}};
Line(6) = {{6, 7}};
Line(7) = {{7, 2}};
Line(8) = {{2, 5}};
Line(10) = {{8, 3}};
Line(11) = {{7, 8}};

Line Loop(1) = {{1, 2, 3, 4}};
Plane Surface(1) = {{1}};
Line Loop(2) = {{8, 5, 6, 7}};
Plane Surface(2) = {{2}};
Line Loop(3) = {{-7, 11, 10, -2}};
Plane Surface(3) = {{3}};

Field[1] = Distance;
Field[1].EdgesList = {{1, 3, 5, 8, 10}};
Field[1].NNodesByEdge = {profile.n_nodes_by_edge};
Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = lc_fine;
Field[2].LcMax = lc_coarse;
Field[2].DistMin = {profile.dist_min};
Field[2].DistMax = {profile.dist_max};
Background Field = 2;

Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0;

// Recombine the swept layer so the 2D surface mesh extrudes into prisms
// instead of letting the 3D mesher tetrahedralize the thin volume.
ext1[] = Extrude {{0, 0, {THICKNESS}}} {{ Surface{{1}}; Layers{{1}}; Recombine; }};
ext2[] = Extrude {{0, 0, {THICKNESS}}} {{ Surface{{2}}; Layers{{1}}; Recombine; }};
ext3[] = Extrude {{0, 0, {THICKNESS}}} {{ Surface{{3}}; Layers{{1}}; Recombine; }};

front1 = ext1[0];
front2 = ext2[0];
front3 = ext3[0];

Physical Volume("internal") = {{ext1[1], ext2[1], ext3[1]}};
Physical Surface("inlet") = {{ext1[5]}};
Physical Surface("Outlet") = {{ext2[4], ext3[3]}};
Physical Surface("fixedWalls") = {{ext1[2], ext1[4], ext2[2], ext2[3], ext3[4]}};
Physical Surface("front") = {{front1, front2, front3}};
Physical Surface("back") = {{1, 2, 3}};

Mesh.Algorithm = 6;
Mesh.ElementOrder = 1;
Mesh.Optimize = 1;
Mesh 3;
"""
    output_path.write_text(geo_content, encoding="utf-8")
    return output_path


def generate_mesh_with_gmsh(geo_file: Path, output_msh: Path | None = None) -> Path:
    geo_file = Path(geo_file)
    output_msh = geo_file.with_suffix(".msh") if output_msh is None else Path(output_msh)
    subprocess.run(
        ["gmsh", "-3", "-format", "msh2", "-o", str(output_msh), str(geo_file)],
        check=True,
    )
    return output_msh
