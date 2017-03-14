%
% Function for computing the anisotropic per-vertex covariances of a PDM
% for a given parameter alpha
%
% Input:
%         alpha            PDM parameter
%         eigVecRel        Modes of variation
%         meanShapeRel     Mean shape
%         uniqueFacesRel   Mesh topology
%         eta              Anisotropy parameter
%
% Output:
%         W                Precision matrix          
%         gradWCell        Partial derivatives of W
%         Winv             Covariance matrix
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
function [W, gradWCell, Winv] = computeAnisotropicCovariances(...
	alpha, eigVecRel, meanShapeRel, uniqueFacesRel, eta)
	persistent allIdx; % we declare this variable as persistent to achieve a caching
	
	gradWCell = [];
	Winv = [];
	
	visualise = 0;
	
	Nrel = size(meanShapeRel,1);
	D = size(meanShapeRel,2);
	M = size(alpha,1);

	if ( isempty(allIdx) || any(size(allIdx) ~= [D*D*Nrel,2]) ) % compute only once
		allIdx = nan(D*D*Nrel,2);
		for i=1:Nrel
			[x,y] = ndgrid(((i-1)*D+1):(i*D),((i-1)*D+1):(i*D));
			allIdx((1:D*D)+(i-1)*(D*D),:) = [x(:), y(:)];
		end
	end
	
	vec = @(x) x(:);
	
	YtransformedRel = reshape(meanShapeRel(:) + eigVecRel*alpha, Nrel, D);
	% PDM normals
	if ( D == 3 )
		y2MinusY1 = YtransformedRel(uniqueFacesRel(:,2),:) - ...
			YtransformedRel(uniqueFacesRel(:,1),:);
		y3MinusY1 = YtransformedRel(uniqueFacesRel(:,3),:) - ...
			YtransformedRel(uniqueFacesRel(:,1),:);
		
		% Calculate normal of face
		b = cross(y2MinusY1,y3MinusY1); % b
		bNorm = sqrt(sum(b.*b,2)); % ||b||_2
		normalsRelNormalised = bsxfun(@rdivide, b, bNorm);
		
		if ( visualise )
			% Show the normals
			fv.vertices = YtransformedRel;
			fv.faces = uniqueFacesRel;
			figure, patch(fv,'FaceColor',[1 0 0]); axis square; hold on;
			for i=1:size(normalsRelNormalised,1);
				p1=fv.vertices(i,:); p2=fv.vertices(i,:)+10*normalsRelNormalised(i,:);
				plot3([p1(1) p2(1)],[p1(2) p2(2)],[p1(3) p2(3)],'g-');
			end
		end
	elseif ( D == 2)
		normalsRelNormalised = nan(Nrel,D);
		for n=1:Nrel
			neighs = uniqueFacesRel(sum(uniqueFacesRel==n,2)>0,:);
			d1 = (YtransformedRel(neighs(1,2),:)-YtransformedRel(neighs(1,1),:));
			d1 = d1./norm(d1);
			d2 = (YtransformedRel(neighs(2,2),:)-YtransformedRel(neighs(2,1),:));
			d2 = d2./norm(d2);
			davg = (d1+d2)/2;
			orth = [-davg(2) davg(1)];
			
			normalsRelNormalised(n,:) = orth./norm(orth);
			if ( visualise )
				quiver(YtransformedRel(n,1), YtransformedRel(n,2), ...
					normalsRelNormalised(n,1), normalsRelNormalised(n,2), 'g', 'Tag', 'PDM');
			end
		end
	end

	% initialise variables
	if ( D == 3 && nargout > 1)
		% compute gradient of normals
		normalsRelGrad = nan(Nrel, D, M);
		for m=1:M
			% \partial n_i = (||b||\partial{b} - b*\partial{||b||}) / ||b||^2
			phi_m = reshape(eigVecRel(:,m), Nrel, D);
				
			phi1_m = phi_m(uniqueFacesRel(:,1),:);
			phi2_m = phi_m(uniqueFacesRel(:,2),:);
			phi3_m = phi_m(uniqueFacesRel(:,3),:);
			
			
			bPartial_m = cross(phi2_m-phi1_m, y3MinusY1) + ...
				cross(y2MinusY1, phi3_m - phi1_m);


			bNormPartial_m = bsxfun(@rdivide, sum(b.*bPartial_m,2), bNorm);
			enum = bsxfun(@times, bPartial_m, bNorm) - ...
				bsxfun(@times, b, bNormPartial_m);
			denom = bNorm.^2;
			normalsRelGrad(:,:,m) = bsxfun(@rdivide, enum, denom);
		end
	end
	

	% compute Wi
	linIdx = 1:D*Nrel;
	unevenIdx = vec(repmat(1:D:D*Nrel, D, 1));
	
	normsVec = vec(normalsRelNormalised');
	Norm = sparse(linIdx, unevenIdx, normsVec, D*Nrel, D*Nrel);
	NNT = Norm*Norm';
	VVT = speye(D*Nrel) - NNT;
	W = eta*NNT + VVT;
	if ( nargout > 2 )
		Winv = (1/eta)*NNT + VVT;
	end

	% compute gradWi
	if ( D == 3 && nargout > 1 )
		gradWCell = cell(M,1);
		for m=1:M
			normsGrad_mVec = vec(normalsRelGrad(:,:, m)');
			
			Ngrad = sparse(linIdx, unevenIdx, normsGrad_mVec, D*Nrel, D*Nrel);
			NgradNorm = Ngrad*Norm';
			gradWCell{m} = (eta-1)*(NgradNorm + NgradNorm');
		end
	end
end
