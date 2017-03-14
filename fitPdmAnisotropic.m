%
% Function for fitting a PDM to points using the method described in [1].
%
% [1] F. Bernard, L. Salamanca, J. Thunberg, A. Tack, D. Jentsch, H. Lamecker,
% S. Zachow, F. Hertel, J. Goncalves, P. Gemmar: Shape-aware Surface 
% Reconstruction from Sparse 3D Point-Clouds. Medical Image Analysis, 38, 
% pp. 77-89, May 2017
% http://dx.doi.org/10.1016/j.media.2017.02.005
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
%
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
%               .doApproximation [1]         toggle approximate Q step, see
%                                            [1] (0 or 1)
%               .convergentApproximation [0] (only applicable if 
%                                            .doApproximation = 1). toggle 
%                                            between doing a convergence
%                                            check or not, see [1] (0 or 1)
%               .singleNewtonStep [0]        toggle between single quasi-Newton 
%                                            step and full quasi-Newton
%                                            method, see [1] (0 or 1)
%               .downsampleProportion [1]    scalar in (0,1] that is a
%                                            downsampling factor. If the
%                                            value is smaller than 1, the
%                                            points are fitted to a
%                                            downsampled version of the
%                                            original mesh. This results in
%                                            faster processing times but
%                                            may also have an impact on
%                                            fitting accuracy.
%               .etaNormalError              error along the normal
%                                            direction (see parameter eta
%                                            in [1]). For performing
%                                            isotropic fitting, set
%                                            etaNormalError to 1. Values
%                                            larger 1 perform anisotropic
%                                            fitting with the error in
%                                            normal direction having larger
%                                            weight than the error in
%                                            tangential direction.
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
function [alpha, alphaHistory, objectiveValue, processingTime, sigma2] = ...
	fitPdmAnisotropic(pdm, sparsePoints, params)
	M = size(sparsePoints,1);
	D = size(sparsePoints,2);
	
	if ( ~isfield(params, 'verbose') )
		verbose = true;
	else
		verbose = params.verbose;
	end
	if ( ~isfield(params, 'doApproximation') )
		doApproximation = 1;
	else
		doApproximation = params.doApproximation;
	end
	if ( ~isfield(params, 'singleNewtonStep') )
		singleNewtonStep = false;
	else
		singleNewtonStep = params.singleNewtonStep;
	end
	if ( D == 3 )
		if ( singleNewtonStep )
			quasiNewtonOptions = optimset('Display','off', ...
				'LargeScale', 'off', ...
				'GradObj' ,'on', 'MaxIter', 1);
		else
			quasiNewtonOptions = optimset('Display','off', ...
				'LargeScale', 'off', ...
				'GradObj' ,'on');
		end
		computeGradientInQ = 1;
	elseif ( D == 2 )
		if ( singleNewtonStep )
			quasiNewtonOptions = optimset('Display','off', ...
				'LargeScale', 'off', ...
				'GradObj' ,'off', 'MaxIter', 1);
		else
			quasiNewtonOptions = optimset('Display','off', ...
				'LargeScale', 'off', ...
				'GradObj' ,'off');
		end
		computeGradientInQ = 0;
	end
	if ( ~isfield(params, 'convergentApproximation') )
		convergentApproximation = 0;
	else
		convergentApproximation = params.convergentApproximation;
	end
	if ( params.etaNormalError == 1 )
		doApproximation = 1; % spherical GMM
	end
	if ( ~isfield(params, 'makeNewFigure') )
		makeNewFigure = 1;
	else
		makeNewFigure = params.makeNewFigure;
	end

	if ( ~isfield(params, 'downsampleProportion') )
		downsampleProportion = 1;
	else
		downsampleProportion = params.downsampleProportion;
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
	
	etaNormalError = params.etaNormalError;

	invCov = diag(1./pdm.eigVal);
	
	if ( nargout > 2 || verbose )
		computeObjective = 1;
	else
		computeObjective = 0;
	end

	meanShape = pdm.meanShape;
	if ( any(size(meanShape) == 1) )
		meanShape = reshape(meanShape, numel(meanShape)/D, D);
	end
	eigVec = pdm.eigVec;

	
	N = size(meanShape,1);
	
	
	if ( ~isfield(pdm, 'relevantPdmIndices') )
		relevantIndices = true(N,1);
	else
		% relevantPdmIndices is used to improve the runtime when using
		% multi-object PDMs where only a subset of the PDM vertices is used
		relevantIndices = pdm.relevantPdmIndices;
	end
	
	if ( D == 3 && (downsampleProportion < 1) )
		if ( any(~relevantIndices) )
			error('Cannot downsample if relevantPdmIndices is used!')
		end
		
		fv.vertices = meanShape;
		fv.faces = pdm.faces;
		
		[fvReduced, idxOfFvVertices] = downsampleMesh(fv, downsampleProportion);
		
		faces = fvReduced.faces;
		
		relevantIndices = false(N,1);
		relevantIndices(idxOfFvVertices) = 1;
		
		% let us find uniqueFaces, such that we have exactly one face per
		% vertex (which is then used for the normal computation)
		repFaces = [faces; faces(:,[2 3 1]); faces(:,[3 1 2])];
		[~,uidx] = unique(repFaces(:,1));
		uniqueFacesRel = repFaces(uidx,:);
	else
		faces = pdm.faces;
		% let us find uniqueFaces, such that we have exactly one face per
		% vertex (which is then used for the normal computation)
		if ( D == 3 )
			repFaces = [faces; faces(:,[2 3 1]); faces(:,[3 1 2])];
			[~,uidx] = unique(repFaces(:,1));
			uniqueFaces = repFaces(uidx,:);
			relIndicesLookup = cumsum(relevantIndices);
			uniqueFacesRel = relIndicesLookup(uniqueFaces(relevantIndices,:));
		elseif ( D == 2 )
			uniqueFacesRel = faces;
		end
	end

	meanShapeRel = meanShape(relevantIndices,:);
	eigVecRel = eigVec(repmat(relevantIndices,D,1),:);
	
	Nrel = size(meanShapeRel,1);
	constraints = ones(Nrel,M);
	if ( isfield(params, 'constraints') )
		constraints = params.constraints(relevantIndices,:);
	end
	
	if ( isfield(params, 'alphaInit') )
		alpha = params.alphaInit;
	else
		alpha = zeros(size(eigVec,2),1);
	end
	alphaOld = inf(size(alpha));
	
	initialShape = meanShapeRel;
	pd = pdist2(initialShape,sparsePoints).^2;
	pd(~constraints) = 0;
	if ( isfield(params, 'sigma2Init') && ~isempty(params.sigma2Init) )
		sigma2 = params.sigma2Init;
	else
		sigma2 = sum(pd(:))/(sum(constraints(:))*D);
	end
	sigma2Old = inf(size(sigma2));
	
	
	alphaHistory = nan(size(eigVec,2), maxIt+1);
	objectiveValue = nan(maxIt+1,1);
	processingTime = nan(maxIt+1,1);
	
	alphaHistory(:,1) = alpha;

	if ( visualise )
		if ( D == 3 )
			if ( downsampleProportion < 1 )
				fvP.vertices = meanShapeRel + reshape(eigVecRel*alpha,Nrel,3);
				fvP.faces = faces;
			else
				fvP.vertices = meanShape + reshape(eigVec*alpha,N,3);
				fvP.faces = pdm.faces;
			end
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
			drawnow;
		elseif ( D == 2 )
			vertices = meanShape + reshape(eigVec*alpha,N,2);
			
			markerSize2d = 24;
			lineWidth2d = 2;
			
			if ( makeNewFigure )
				figure('Color', 'k','Position', [900 0 900 467]);
			else
				figure(1);
				clf;
				set(gcf, 'Color', 'k');
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

	
	
	if ( D == 3 )
		Perm = [kron(speye(Nrel),[1 0 0]); ...
			kron(speye(Nrel),[0 1 0]); ...
			kron(speye(Nrel),[0 0 1])];
	elseif ( D == 2 )
		Perm = [kron(speye(Nrel),[1 0]); ...
			kron(speye(Nrel),[0 1])];
	end
			
	
	iterTic = tic();
	
	iter = 0; 
	
	% initialise Y and W
	YtransformedRel3 = reshape(meanShapeRel(:) + eigVecRel*alpha, Nrel, D);
	W = computeAnisotropicCovariances(alpha, eigVecRel, meanShapeRel, ...
		uniqueFacesRel, etaNormalError);
	SigmaBig = Perm*W*Perm';

	%% USE ANISOTROPIC FITTING
	while ( iter < maxIt && norm(alpha-alphaOld) > paramTolerance && ...
			norm(sigma2-sigma2Old) > paramTolerance )
		
		alphaOld = alpha;
		sigma2Old = sigma2;
		
		%% E-STEP
		Pmn = zeros(M,Nrel);
		for m=1:M % for each n do
			% enumerator
			diffVec = bsxfun(@minus, sparsePoints(m,:), YtransformedRel3)';
			diffVec = diffVec(:);
			squaredDistance = sum(reshape((diffVec'*W)'.*diffVec, D, Nrel)',2);
			
			P = constraints(:,m).*exp(...
				squaredDistance/(-2*sigma2));
			Pmn(m,:) = P./sum(P);
		end
		
		%% Compute Objective
		if ( computeObjective && iter == 0)
			Q = computeAnisotropicObjective(alpha, eigVecRel, meanShapeRel, ...
				sigma2, uniqueFacesRel, etaNormalError, Pmn, invCov, ...
				sparsePoints, regulariseFlag);
			objectiveValue(1) = Q;
			processingTime(1) = 0;
			if ( verbose )
				disp(['Q = ' num2str(Q) ' iter = ' num2str(iter) ' |delta-alpha| = ' ...
					num2str(norm(alpha-alphaOld)) ' |delta-sigma2| = '  num2str(norm(sigma2-sigma2Old)) ' sigma2 = ' num2str(sigma2)]);
			end
		end
		
		%% M-STEP (vectorised)
		P = kron(eye(D),Pmn);
		oneMT = ones(1,D*M);
		diag1MTP = sparse(1:size(P,2),1:size(P,2), oneMT*P);
		
		%% alpha-step
		if ( doApproximation ) % ANISO method
			twoeigSig = 2*eigVecRel'*SigmaBig';
			A = twoeigSig*diag1MTP*eigVecRel;
			b = twoeigSig*P'*sparsePoints(:) - ...
				eigVecRel'*SigmaBig'*diag1MTP*meanShapeRel(:) - ...
				eigVecRel'*diag1MTP'*SigmaBig*meanShapeRel(:);
			
			if ( regulariseFlag )
				A = A + 2*sigma2*invCov;
			end
			alphaNew = A\b;
			
			if ( convergentApproximation ) % ANISOc method
				oldObj = computeAnisotropicObjective(alpha, ...
					eigVecRel, meanShapeRel, sigma2, uniqueFacesRel, etaNormalError,...
					Pmn, invCov, ...
					sparsePoints, regulariseFlag);
				newObj = computeAnisotropicObjective(alphaNew, ...
					eigVecRel, meanShapeRel, sigma2, uniqueFacesRel, etaNormalError,...
					Pmn, invCov, ...
					sparsePoints, regulariseFlag);
				if ( newObj < oldObj ) % run quasi-Newton method to fix problem
					objFun = @(currAlpha) computeAnisotropicObjective(currAlpha, ...
						eigVecRel, meanShapeRel, sigma2, uniqueFacesRel, etaNormalError,...
						Pmn, invCov, ...
						sparsePoints, regulariseFlag, computeGradientInQ, true);
					
					alphaNew = fminunc(objFun, alpha, quasiNewtonOptions);
				end
			end
			alpha = alphaNew;
		else %  use quasi-Newton method (ECM or GEM, depending on params)
			objFun = @(currAlpha) computeAnisotropicObjective(currAlpha, ...
				eigVecRel, meanShapeRel, sigma2, uniqueFacesRel, etaNormalError,...
				Pmn, invCov, ...
				sparsePoints, regulariseFlag, computeGradientInQ, true);

			alpha = fminunc(objFun, alpha, quasiNewtonOptions);
		end
		
		%% sigma-step
		W = computeAnisotropicCovariances(alpha, eigVecRel, meanShapeRel, ...
			uniqueFacesRel, etaNormalError);
		SigmaBig = Perm*W*Perm';
		
		if ( D == 3 )
			ptSigmaJp = (kron(eye(D), Pmn)*diag(SigmaBig))'* ...
				(sparsePoints(:).^2) + ...
				2*(sparsePoints(M+1:end).*sparsePoints(1:2*M))* ...
				(kron(eye(D-1), Pmn)*diag(SigmaBig(Nrel+1:end,1:2*Nrel))) + ...
				2*(sparsePoints(2*M+1:end).*sparsePoints(1:M))* ...
				(kron(eye(D-2), Pmn)*diag(SigmaBig(2*Nrel+1:end,1:Nrel)));
		elseif ( D == 2 )
			ptSigmaJp = (kron(eye(D), Pmn)*diag(SigmaBig))'* ...
				(sparsePoints(:).^2) + ...
				2*(sparsePoints(M+1:end).*sparsePoints(1:M))* ...
				(kron(eye(D-1), Pmn)*diag(SigmaBig(Nrel+1:end,1:2*Nrel)));
		end
		
		YTransformedRel = meanShapeRel(:) + eigVecRel*alpha;
		YtransformedRel3 = reshape(YTransformedRel, Nrel, D);

		sigma2 = (ptSigmaJp - ...
			2*sparsePoints(:)'*P*SigmaBig*YTransformedRel + ...
			YTransformedRel'*SigmaBig'*diag1MTP*YTransformedRel)/(M*D);
		
		% end of M-step
		if ( visualise )
			delete(findall(gcf,'Tag', 'PDM'));
			if ( D == 3 )
				if ( downsampleProportion < 1 )
					fvP.vertices = meanShapeRel + reshape(eigVecRel*alpha,Nrel,3);
					fvP.faces = faces;
				else
					fvP.vertices = meanShape + reshape(eigVec*alpha,N,3);
					fvP.faces = pdm.faces;
				end
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
		
		iter = iter + 1;
		
		alphaHistory(:,iter+1) = alpha;
		if ( computeObjective )
			Q = computeAnisotropicObjective(alpha, eigVecRel, meanShapeRel, ...
				sigma2, uniqueFacesRel, etaNormalError, Pmn, invCov, ...
				sparsePoints, regulariseFlag);
			objectiveValue(iter+1) = Q;
			processingTime(iter+1) = toc(iterTic);
			
			if ( verbose )
				disp(['Q = ' num2str(Q,10) ' iter = ' num2str(iter) ' |delta-alpha| = ' ...
					num2str(norm(alpha-alphaOld)) ' |delta-sigma2| = '  num2str(norm(sigma2-sigma2Old)) ' sigma2 = ' num2str(sigma2)]);
			end
		end
	end
	
	if ( visualise )
		hold off;
	end
	
	% check if Q has decreased (up to numerical precision)
	if ( any( diff(objectiveValue(~isnan(objectiveValue))) < -paramTolerance ) )
		iters = find(diff(objectiveValue(~isnan(objectiveValue))) < 0);
		if ( doApproximation && ~convergentApproximation )
			warning(['Q has decreased at iterations ' num2str((iters)')]);
		else % in thise case we must have convergence and thus raise an exception
			error(['Q has decreased at iterations ' num2str((iters)')]);
		end
	end
	
end