Template-matching geometries are stored in [subTomoMeta][cycle000][geometry] in an array with 26 columns.
Several of these columns are simply empty or default values (according to the emC tutorial)
0: CC index
1: Tmp_sampling (current binning of image)
2: empty
3: Unique particle ID
4-9: empty
10-12: x, y, z coordinates describing centre of particle
13-15: ϕ, θ, ψ euler angles describing particle orientation in intrinsic z-x-z convention. Not actually used by emClarity
16-24: 3D Rotation matrix describing particle orientation. Ordered 11, 21, 31, 12, 22, 32, 13, 23, 33
25: Particle classification. Should be 1 at this stage.
