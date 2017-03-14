%
% Function for fitting a PDM to points using regularised ICP.
%
% Input:
%         pdm          Point distribution model (PDM) struct with the
%                      following fields
%
%               .meanShape:   vector of dimension 3*N, containing the mean 
%                             of all shapes. The first N elements correspond 
%                             to the x components, followed by N elements 
%                             for the y components, and N elements for the
%                             z components
%               .eigVec:      3*N x M matrix, containing in each column one 
%                             mode of variation
%               .eigVal:      vector of dimension M, containing the variance 
%                             of each mode of variation
%               .faces:       nFaces x 3 matrix, containing nFaces triangular 
%                             faces, where each row contains three integers 
%                             between 1 and N.
%         sparsePoints nPts x 3 matrix containing the points that we want
%                      to fit the PDM to
%         params       Structure containing the following parameters (optional)
%
%               .paramTolerance [1e-5]       tolerance for convergence
%               .regulariseFlag [1]          toggle regularisation (0 or 1)
%               .visualise [1]               toggle visualisation (0 or 1)
%               .maxIt [100]                 maximum number of iterations
%               .verbose [1]                 toggle verbose (0 or 1)
%               .makeNewFigure [1]           toggle creating a new figure
%                                            (0 or 1)
%               .constraints [ones(N,nPts)]  constraints(i,j) indicates
%                                            whether it is allowed that 
%                                            the i-th PDM point is matched 
%                                            with the j-th point (0 or 1)
%               .alphaInit [zeros(M,1)]      initial value of alpha
%      
%
% Output: 
%       alpha:          PDM parameter that best fits the PDM to the
%                       sparsePoints
%       alphaHistory:   alphaHistory(:,i) denotes alpha at iteration (i-1)
%       objectiveValue: objectiveValue(i) denotes the value of the objective at iteration (i-1)
%                       (for the probabilistic fitting procedure it is the
%                       value of the Q function, and for icp it is squared error)
%       processingTime: sum of total processing time in seconds (per iteration)
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
function [alpha, alphaHistory, objectiveValue, processingTime] = ...
	fitPdmIcp(pdm, sparsePoints, params)
	nPts = size(sparsePoints,1);
	D = size(sparsePoints,2);
	if ( ~exist('params', 'var') )
		params = [];
	end
	if ( ~isfield(params, 'verbose') )
		verbose = true;
	else
		verbose = params.verbose;
	end
	if ( ~isfield(params, 'makeNewFigure') )
		makeNewFigure = 1;
	else
		makeNewFigure = params.makeNewFigure;
	end
	if ( ~isfield(params, 'maxIt') )
		maxIt = 100;
	else
		maxIt = params.maxIt;
	end
	if ( ~isfield(params, 'regulariseFlag') )
		regulariseFlag = 1;
	else
		regulariseFlag = params.regulariseFlag;
	end
	if ( ~isfield(params, 'paramTolerance') )
		paramTolerance = 1e-5;
	else
		paramTolerance = params.paramTolerance;
	end
	if ( ~isfield(params, 'visualise') )
		visualise = 1;
	else
		visualise = params.visualise;
	end

	if ( nargout > 2 || verbose )
		computeObjective = 1;
	else
		computeObjective = 0;
	end
	
	invCov = diag(1./pdm.eigVal);
	

	meanShape = pdm.meanShape;
	if ( size(meanShape,2) == 1 )
		meanShape = reshape(meanShape, numel(meanShape)/D, D);
	end
	eigVec = pdm.eigVec;
	
	N = size(meanShape,1);
	if ( ~isfield(pdm, 'relevantPdmIndices') )
		relevantIndices = true(N,1);
	else
		relevantIndices = pdm.relevantPdmIndices;
	end

	meanShapeRel = meanShape(relevantIndices,:);
	eigVecRel = eigVec(repmat(relevantIndices,D,1),:);
	
	Nrel = size(meanShapeRel,1);
	
	constraints = ones(Nrel,nPts);
	if ( isfield(params, 'constraints') )
		constraints = params.constraints(relevantIndices,:);
	end
	
	if ( isfield(params, 'alphaInit') )
		alpha = params.alphaInit;
	else
		alpha = zeros(size(eigVec,2),1);
	end
	alphaOld = inf(size(alpha));
	
	alphaHistory = nan(size(eigVec,2), maxIt+1);
	objectiveValue = nan(maxIt+1,1);
	processingTime = nan(maxIt+1,1);
	

	if ( visualise )
		if ( D == 3 )
			fvP.vertices = meanShape + reshape(eigVec*alpha,N,3);
			fvP.faces = pdm.faces;
			if ( makeNewFigure )
				figure('Color', 'k','Position', [900 0 900 700]);
			else
				figure(1);
				clf;
				set(gcf, 'Color', 'k'); %,'Position', [900 0 900 700]);
			end
			ax = axes('Color', 'k', 'Position', [0 0 1 1]);
			axis off;
			axis equal;
			
			hold on;
			patch(fvP, 'FaceAlpha', 0.5, 'FaceColor', 'g', 'Tag', 'PDM', 'EdgeColor', 'w', 'LineWidth', 2);
		
			view([pi, -90])
			
			plot3(sparsePoints(:,1),sparsePoints(:,2),sparsePoints(:,3), '.r', 'MarkerSize', 32),
			axis tight;
			cameratoolbar('SetMode','orbit')
		elseif ( D == 2 )
			vertices = meanShape + reshape(eigVec*alpha,N,2);
			
			markerSize2d = 24;
			lineWidth2d = 2;
			
			if ( makeNewFigure )
				figure('Color', 'k','Position', [900 0 900 467]);
			else
				figure(1);
				clf;
				set(gcf, 'Color', 'k'); %,'Position', [900 0 900 700]);
			end
			ax = axes('Color', 'k', 'Position', [0 0 1 1]);
			axis off;
			axis equal;
			
			hold on;
			hold on;

			plot(sparsePoints(:,1),sparsePoints(:,2), '.r', ...
				'MarkerSize', markerSize2d),

			minpts = min(vertices);
			maxpts = max(vertices);
			xlim([minpts(1)-.4 maxpts(1)+.4]);
			ylim([minpts(2)-.4 maxpts(2)+.4]);
		end
	end
			
	iterTic = tic();
	iter=0; 
	
	disp('Running ICP Fitting...');
	
	alphaHistory(:,1) = alpha;
	if ( computeObjective )
		YtransformedRel = reshape(meanShapeRel(:) + eigVecRel*alpha, Nrel, D);
		
		% find correspondences (considering constraints)
		pd = pdist2(YtransformedRel, sparsePoints);
		pd(~constraints) = inf;
		[~,idxXyz] = min(pd);
		idxXyz = idxXyz';
		
		if ( D == 3 )
			idxCol = [idxXyz, idxXyz + Nrel,idxXyz + 2*Nrel];
		elseif ( D == 2 )
			idxCol = [idxXyz, idxXyz + Nrel];
		end
		
		objectiveValue(1) = ...
			norm(vec(sparsePoints - meanShapeRel(idxXyz,:)) - ...
			(eigVecRel(idxCol,:)*alpha),2).^2;
		
		processingTime(1) = 0;
		
		if ( verbose )
			disp(['error = ' num2str(objectiveValue(1)) ' iter = ' num2str(iter) ' |delta-alpha| = ' ...
				num2str(norm(alpha-alphaOld))]);
		end
	end

	%% ICP method
	while ( iter < maxIt && ...
			norm(alpha-alphaOld) > paramTolerance )
		alphaOld = alpha;

		YtransformedRel = reshape(meanShapeRel(:) + eigVecRel*alpha, Nrel, D);

		% find correspondences (considering constraints)
		pd = pdist2(YtransformedRel, sparsePoints);
		pd(~constraints) = inf;
		[~,idxXyz] = min(pd);
		idxXyz = idxXyz';
		
		if ( D == 3 )
			idxCol = [idxXyz, idxXyz + Nrel,idxXyz + 2*Nrel];
		elseif ( D == 2 )
			idxCol = [idxXyz, idxXyz + Nrel];
		end
		
		if ( visualise )
			delete(findall(gcf,'Tag', 'PDM'));
			if ( D == 3 )
				fvP.vertices = meanShape + reshape(eigVec*alpha,N,3);
				fvP.faces = pdm.faces;
				patch(fvP, 'FaceAlpha', 0.5, 'FaceColor', 'g', 'Tag', ...
					'PDM', 'EdgeColor', 'w', 'LineWidth', 2);
			elseif ( D == 2)
				vertices = meanShape + reshape(eigVec*alpha,N,2);
				
				hold on;
				plot([vertices(:,1); vertices(1,1)], [vertices(:,2); vertices(1,2)], ...
					'g', 'LineWidth', lineWidth2d, 'Tag', 'PDM');
				hold on;
				plot(vertices(:,1), vertices(:,2), 'g.', 'MarkerSize', markerSize2d,'Tag', 'PDM');
				plot(sparsePoints(:,1),sparsePoints(:,2), '.r', 'MarkerSize', markerSize2d),
			end
			drawnow;
		end
		
		% find alpha
		b =  sparsePoints - meanShapeRel(idxXyz,:);
		if ( regulariseFlag )
			% use tikhonov regularisation
			A = eigVecRel(idxCol,:);

			alpha = (A'*A + invCov)\A'*b(:);
		else
			A = eigVecRel(idxCol,:);
			alpha = A\b(:);
		end
		
		iter = iter + 1;
		
		alphaHistory(:,iter+1) = alpha;
		if ( computeObjective )
			objectiveValue(iter+1) = ...
				norm(vec(sparsePoints - meanShapeRel(idxXyz,:)) - ...
				(eigVecRel(idxCol,:)*alpha),2).^2;
			processingTime(iter+1) = toc(iterTic);
			if ( verbose )
				disp(['error = ' num2str(objectiveValue(iter+1)) ' iter = ' num2str(iter) ' |delta-alpha| = ' ...
					num2str(norm(alpha-alphaOld))]);
			end
		end
	end
	
	
	if ( visualise )
		hold off;
	end
	
	if ( any( diff(objectiveValue(~isnan(objectiveValue))) > 0 ) )
		iters = find(diff(objectiveValue(~isnan(objectiveValue))) > 0);
		
		warning(['error has increased at iterations ' num2str((iters)')]);
	end
	
end