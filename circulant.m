function V = circulant(V, d)
% CIRCULANT - Circulant Matrix
%
%   C = CIRCULANT(V) or CIRCULANT(V, 1) returns the circulant matrix C
%   based on the row/column vector V. C is a square matrix in which each
%   row/column is a formed by circularly shifting the preceeding row/column
%   forward by one element. The first row (or column) is equal to V.
%
%   C = CIRCULANT(V, -1) applies a backward shift, returning a symmetric
%   matrix, so that C equals TRANSPOSE(C). In this case, it does not matter
%   if V is a row or column vector.
%
%   Examples:
%      circulant([2 3 5]) % forward shift
%        % ->   2     3     5
%        %      5     2     3
%        %      3     5     2
%
%      circulant([2 3 5].') % column input
%        % ->   2     5     3
%        %      3     2     5
%        %      5     3     2
%
%      circulant([2 3 5], -1),  circulant([2 ; 3  ; 5], -1) % backward shift
%      % for both row or column vectors, this returns a symmetric matrix:
%        % ->   2     3     5
%        %      3     5     2
%        %      5     2     3
%
%  The output has the same type as the input.
%
%      V =  {'One'  '2'  'III'}
%      circulant(A)
%        % ->      'One'    '2'      'III'
%        %         'III'    'One'    '2' 
%        %         '2'      'III'    'One' 
%
%   Notes:
%   - This version is completely based on indexing and does not use loops,
%     repmat, hankel, toeplitz or bsxfun. It should therefore run pretty
%     fast on most Matlab versions.
%   - See http://en.wikipedia.org/wiki/Circulant_matrix for more info on
%     circulant matrices.
%
%   See also TOEPLITZ, HANKEL
%            LATSQ, BALLATSQ (on the File Exchange)
% for Matlab R13 and up
% version 2.2 (feb 2019)
% (c) Jos van der Geest
% email: samelinoa@gmail.com
% History
% 1.0 (feb 2009) - 
% 2.0 (feb 2009) - Important bug fix for row vectors with forward shift
% 2.1 (may 2016) - updated for recent ML versions
% 2.2 (feb 2019) - modernised
% Acknowledgements:
% This file was inspired by two submissions on the File Exchange (#22814 &
% #22858). I modified some of the help from the submission by John D'Errico.
N = numel(V) ; 
if N > 0 && ~isvector(V)
    error('Input should be a vector') ;
end
if nargin==2
    if ~(isequal(d, -1) || isequal(d, 1))
        error(['Second argument should either be 1 (forward\n  shift, ' ...
            'default) or -1 (backward shift).']) ;
    end
else
    d = 1 ; 
end
% for an empty matrix or single element there is nothing to do.
if N > 1       
    % Create a circulant index matrix using cumsum and rem:    
    idx = repmat(-d, N, N) ;     % takes care of forward or backward shifts
    idx(:,1) = 1:N ;             % creating the shift ..
    idx = cumsum(idx, 2) ;       % .. by applying cumsum
    idx = rem(idx+N-1, N) + 1 ;  % all idx become positive by adding N first
    
    if d==1 && size(V,1)==1    
        % needed for row vectors with forward shift (bug fixed in v2.0)
        idx = idx.' ;
    end
    
    V = V(idx) ;
end
