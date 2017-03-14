%
% Function for downsampling a surface mesh. For that, a subset of the
% original mesh vertices is selected and remeshed.
%
% Input:
%         fv         Matlab fv structure (cf. Matlab's isosurface()/patch() functions)
%                    that represents the triangular surface mesh
%         factor     downsampling factor, scalar value in (0,1] 
%
% Output:
%         fvReduced  Downsampled Matlab fv structure
%         fvIdx      Vertex indices of the original fv structure that are
%                    retained
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
function [fvReduced, fvIdx] = downsampleMesh(fv, factor)
	vertices = fv.vertices;

	fvReduced = reducepatch(fv,factor);

	reducedVertices = fvReduced.vertices;
	[idxOfFvVertices,d] = knnsearch(vertices, reducedVertices);
	assert(sum(d)==0);
	
	% recreate original vertex order
	[fvIdx, idx] = sort(idxOfFvVertices);
	reducedVerticesSorted = reducedVertices(idx,:);
	
	[~,tmpidx] = sort(idx);
	
	facesSorted = tmpidx(fvReduced.faces);
	fvReduced.vertices = reducedVerticesSorted;
	fvReduced.faces = facesSorted;
	
	assert(sum(d)==0); % make sure vertices are not moved
end
