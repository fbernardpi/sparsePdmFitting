% Demo of sparse SSM fitting procedure for ellipsoid pdm
%
% Author & Copyright (C) 2017: Florian Bernard (f.bernardpi[at]gmail[dot]com)
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU Affero General Public License as published
% by the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.

% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU Affero General Public License for more details.

% You should have received a copy of the GNU Affero General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
%

clear;
close all;

%% LOAD DATA
load ellipsoids_pdm.mat

N = size(pdm.eigVec,1)/3; % numer of vertices
M = size(pdm.eigVec,2); % number of modes


%% RANDOMLY DRAW POINTS ON SHAPE SURFACE
P = 9; % number of sparse points
sigmaNoise = 0.01; % noise of sparse points
visualiseRandomSamples = 1; % flag for visualisation

% draw an alpha from a Gaussian distribution
alphaGt = mvnrnd(zeros(M,1), diag(pdm.eigVal))';

% create shape for the given alpha
v = reshape(pdm.meanShape(:) + pdm.eigVec*alphaGt, N, 3);

% create Matlab fv structure (triangular surface mesh) of the sampled shape
fv.faces = pdm.faces;
fv.vertices = v;

% select P points uniformly from surface
sparsePoints = sampleMeshSurfacePointsUniformly(fv, P, visualiseRandomSamples, sigmaNoise);

%% ANISO
paramsAniso = [];
paramsAniso.etaNormalError = 2; % importance of error along normal

alphaAniso = fitPdmAnisotropic(pdm, sparsePoints, paramsAniso);

%% ISO
paramsIso = [];
paramsIso.etaNormalError = 1; 

alphaIso = fitPdmAnisotropic(pdm, sparsePoints, paramsIso);

%% ICP-based fitting
alphaIcp = fitPdmIcp(pdm, sparsePoints);

%% compare ANISO and ICP
disp(['Error of ANISO = ' num2str(norm(alphaGt-alphaAniso))]);
disp(['Error of ISO = ' num2str(norm(alphaGt-alphaIso))]);
disp(['Error of ICP = ' num2str(norm(alphaGt-alphaIcp))]);

