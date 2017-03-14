%
% Function for uniformly sampling points on a triangular surface mesh.
%
% Input:
%         fv         Matlab fv structure (cf. Matlab's isosurface()/patch() functions)
%                    that represents the triangular surface mesh
%         nPts       Number of surface points that shall be sampled
%         visualise  Flag for enabling/disabling visualisation (values 0 or
%                    1)
%         sigma      Variance of Gaussian noise that is added to the surface points
%
% Output:
%         pts        nPts x 3 matrix containing the sampled surface points
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
function pts = sampleMeshSurfacePointsUniformly(fv, nPts, visualise, sigma)
	if ( ~exist('visualise', 'var') )
		visualise = 0;
	end
	if ( ~exist('sigma', 'var') )
		sigma = 0;
	end
	% use heron's formula to compute triangle area, see 
	% https://en.wikipedia.org/wiki/Heron's_formula
	vertices = fv.vertices;
	faces = fv.faces;
	
	sideA = vertices(faces(:,1),:) - vertices(faces(:,2),:);
	sideB = vertices(faces(:,2),:) - vertices(faces(:,3),:);
	sideC = vertices(faces(:,3),:) - vertices(faces(:,1),:);
	
	a = sqrt(sum(sideA.*sideA,2));
	b = sqrt(sum(sideB.*sideB,2));
	c = sqrt(sum(sideC.*sideC,2));
	s = (a+b+c)./2;
	
	area = sqrt(s.*(s-a).*(s-b).*(s-c));
	areaNorm = area./sum(area);
	cumsumArea = [0; cumsum(areaNorm)];
	
	rndNumber = rand(nPts,1);
	rndTriangleIdx = sum(bsxfun(@ge, rndNumber, cumsumArea'),2);
	
	% figure, hist(rndTriangleIdx, size(faces,1))
	
	% sample uniformly from triangle, see
	% http://www.cs.princeton.edu/~funk/tog02.pdf, section 4.2
	A = vertices(faces(rndTriangleIdx,1),:);
	B = vertices(faces(rndTriangleIdx,2),:);
	C = vertices(faces(rndTriangleIdx,3),:);
	
	r1 = rand(nPts,1);
	r2 = rand(nPts,1);
	ptsUnnoisy = bsxfun(@times, 1 - sqrt(r1), A) + ...
		bsxfun(@times, sqrt(r1).*(1-r2), B) + ...
		bsxfun(@times, sqrt(r1).*r2, C);
	
	% add Gaussian noise
	pts = ptsUnnoisy + randn(size(ptsUnnoisy,1),3)*sigma;
	
% 	figure, plot3([A(1,1) B(1,1) C(1,1) A(1,1)], [A(1,2) B(1,2) C(1,2) A(1,2)], ...
% 		[A(1,3) B(1,3) C(1,3) A(1,3)]);
% 	hold on;
% 	plot3(pts(1,1),pts(1,2),pts(1,3), 'r.');
% 	rotate3d;
	
	if ( visualise )
		figure('Color', 'k','Position', [900 0 900 700]);
		
		ax = axes('Color', 'k', 'Position', [0 0 1 1]);
		axis off;
		axis equal;
		
		hold on;
		patch(fv, 'FaceAlpha', 0.5, 'FaceColor', [1 1 1].*0.4, 'Tag', 'PDM', 'EdgeColor', 'w', 'LineWidth', 2);
		
		view([pi, -90])
		axis tight;

		plot3(pts(:,1), pts(:,2), pts(:,3), '.g', 'MarkerSize', 32, 'Tag', 'ptsNoisy');
		plot3([ptsUnnoisy(:,1) pts(:,1)]', ... 
			[ptsUnnoisy(:,2) pts(:,2)]', ...
			[ptsUnnoisy(:,3) pts(:,3)]', 'g', 'LineWidth', 4, 'Tag', 'ptsNoisy');
		plot3(ptsUnnoisy(:,1), ptsUnnoisy(:,2), ptsUnnoisy(:,3), '.r', 'MarkerSize', 32, 'Tag', 'pts');
		cameratoolbar('SetMode','orbit')
	end
end