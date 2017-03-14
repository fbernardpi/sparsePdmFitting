%
% Function for computing the anisotropic objective function.
%
% Input:
%         alpha            PDM parameter
%         eigVecRel        Modes of variation
%         meanShapeRel     Mean shape
%         sigma2           Value of sigma^2
%         uniqueFacesRel   Mesh topology
%         eta              Anisotropy parameter
%         Pmn              Probabilistic correspondences computed in E-step
%         alphaCovInv      Inverse of covariance matrix of alpha (which is
%                          assumed to have a zero-mean Gaussian distribution)
%         sparsePoints     Sparse points that the PDM is fitted to
%         regulariseFlag   Toggle regularisation
%         computeGradient  Toggle computation of gradient
%         minusObjective   Flip the sign of the objective value (useful for
%                          using Matlab's fminunc function to maximise the
%                          objective)
%         alphaForCov      alpha that is used to compute the anisotropi
%                          covariances
%
% Output:
%         Q                Objective value (Q function)        
%         gradQalpha       Gradient of Q
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

function [Q,gradQalpha] = computeAnisotropicObjective(alpha, eigVecRel, meanShapeRel, ...
	sigma2, uniqueFacesRel, eta, Pmn, alphaCovInv, sparsePoints, regulariseFlag, ...
	computeGradient, minusObjective, alphaForCov)

	persistent Perm; % we declare this variable as persistent to achieve a caching
	
	
	Nrel = size(meanShapeRel,1);
	D = size(meanShapeRel,2);
	M = size(sparsePoints,1);

	if ( ~exist('computeGradient', 'var') )
		computeGradient = 0;
	end
	if ( ~exist('minusObjective', 'var') )
		minusObjective = 0;
	end
	if ( ~exist('alphaForCov', 'var') )
		alphaForCov = alpha; % this can be used to compute Qtilde
	end
	if ( computeGradient )
		[W,gradWCell] = computeAnisotropicCovariances(...
			alphaForCov, eigVecRel, meanShapeRel, ...
			uniqueFacesRel, eta);
	else
		W = computeAnisotropicCovariances(...
			alphaForCov, eigVecRel, meanShapeRel, ...
			uniqueFacesRel, eta);
	end
				
	YtransformedRel = reshape(meanShapeRel(:) + eigVecRel*alpha, Nrel, D);

	if ( isempty(Perm) || any(size(Perm) ~= [Nrel*D, Nrel*D]) )
		if ( D == 3 )
			Perm = [kron(speye(Nrel),[1 0 0]); ...
				kron(speye(Nrel),[0 1 0]); ...
				kron(speye(Nrel),[0 0 1])];
		elseif ( D == 2 )
			Perm = [kron(speye(Nrel),[1 0]); ...
				kron(speye(Nrel),[0 1])];
		end
	end
	SigmaBig = Perm*W*Perm';
	
	kronEyeDPmn = kron(eye(D), Pmn);
	kronEyeDMinusOnePmn = kron(eye(D-1), Pmn);
	sparsePointsSquareAA = (sparsePoints(:).^2);
	if ( D == 3 )
		kronEyeDMinusTwoPmn = kron(eye(D-2), Pmn);
		
		twoSparsePointsSquareBC = 2*(sparsePoints(M+1:end).*sparsePoints(1:2*M));
		twoSparsePointsSquareCA = 2*(sparsePoints(2*M+1:end).*sparsePoints(1:M));
		
		BCtimesKron = twoSparsePointsSquareBC*kronEyeDMinusOnePmn;
		CAtimesKron = twoSparsePointsSquareCA*kronEyeDMinusTwoPmn;
		
		ptSigmaJp = (kronEyeDPmn*diag(SigmaBig))'*sparsePointsSquareAA + ...
			BCtimesKron*diag(SigmaBig(Nrel+1:end,1:2*Nrel)) + ...
			CAtimesKron*diag(SigmaBig(2*Nrel+1:end,1:Nrel));
	elseif ( D == 2 )
		BAtimesKron = 2*(sparsePoints(M+1:end).*sparsePoints(1:M))*kronEyeDMinusOnePmn;

		ptSigmaJp = (kron(eye(D), Pmn)*diag(SigmaBig))'*sparsePointsSquareAA + ...
			BAtimesKron*diag(SigmaBig(Nrel+1:end,1:2*Nrel));
	end
	
	P = kron(eye(D),Pmn);
	oneMT = ones(1,D*M);
	diag1MTP = sparse(1:size(P,2),1:size(P,2), oneMT*P);
	diag1MTPY = diag1MTP*YtransformedRel(:);
			
	SigmaBigDiag1MTPY = SigmaBig'*diag1MTPY;
	twoSparsePointsTimesP = 2*sparsePoints(:)'*P;
	twoSparsePointsTimesPTimesSigmaBig = twoSparsePointsTimesP*SigmaBig;
	
	Q = ptSigmaJp - twoSparsePointsTimesPTimesSigmaBig*YtransformedRel(:) + ...
		YtransformedRel(:)'*SigmaBigDiag1MTPY;
	
	%% compute gradient
	if ( computeGradient )
		%% precompute some variables in order not do it repeatedly in for-loop
		constTerm = 2*eigVecRel'*SigmaBigDiag1MTPY -...
			(twoSparsePointsTimesPTimesSigmaBig*eigVecRel)';
		
		%% compute partial derivatives
		gradQalpha = zeros(size(alpha));
		for a=1:size(alpha,1) 
			gradW_mPerm = Perm*gradWCell{a}*Perm';
			if ( D == 3 )
				ptGradWp = (kronEyeDPmn*diag(gradW_mPerm))'*sparsePointsSquareAA + ...
					BCtimesKron*diag(gradW_mPerm(Nrel+1:end,1:2*Nrel)) + ...
					CAtimesKron*diag(gradW_mPerm(2*Nrel+1:end,1:Nrel));
			elseif ( D == 2 )
				ptGradWp = (kronEyeDPmn*diag(SiggradW_mPermmaBig))'* ...
					sparsePointsSquareAA + ...
					BAtimesKron*diag(gradW_mPerm(Nrel+1:end,1:2*Nrel));
			end
		
			gradQalpha(a) = ptGradWp - ...
				twoSparsePointsTimesP*gradW_mPerm*YtransformedRel(:) + ...
				YtransformedRel(:)'*gradW_mPerm'*diag1MTPY;
		end
		gradQalpha = gradQalpha + constTerm;
		gradQalpha = -gradQalpha;
	end

	Q = -(D/2)*M*log(sigma2) - Q/(2*sigma2);
	Q = Q - sum(Pmn(Pmn(:)>0).*log(Pmn(Pmn(:)>0)));
	
	if ( regulariseFlag )
		Q = Q - 0.5 * alpha'*alphaCovInv*alpha;
	end
	
	if ( minusObjective )
		Q = -Q; % useful for matlab's fminunc function
	end
	
	if ( computeGradient ) % compute gradient
		% compute gradient direction
		gradQalpha = gradQalpha./(2*sigma2);
		if ( regulariseFlag )
			gradQalpha = gradQalpha - alphaCovInv*alpha;
		end
			
		if ( minusObjective )
			gradQalpha = -gradQalpha; % useful for matlab's fminunc function
		end
	end
end
