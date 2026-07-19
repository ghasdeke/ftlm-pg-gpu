function test_kagome_spacegroup()
%TEST_KAGOME_SPACEGROUP  Verify the kagome torus space-group provider.
%   N=12 (|G|=48) and N=36 (|G|=144): order, NN bonds (2N, degree 4), group
%   axioms (Latin-square mul, inverses), bond invariance under every element,
%   and MIN_IMAGE_IH consistency vs brute force.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    rng(7);
    cases = [2 0; 2 2];                 % N=12 (|G|=48), N=36 (|G|=144)
    for ci = 1:size(cases,1)
        a=cases(ci,1); b=cases(ci,2);
        [g, bonds] = kagome_spacegroup(a,b);
        N=g.N; fprintf('=== kagome (a,b)=(%d,%d): N=%d, |G|=%d, %d bonds ===\n', a,b,N,g.order,size(bonds,1));
        check(g.order == 12*g.ncells, 'order');
        check(size(bonds,1) == 2*N, 'bond count');
        deg = accumarray([bonds(:,1);bonds(:,2)],1,[N 1]);
        check(all(deg==4), 'every site degree 4 (got min %d max %d)', min(deg), max(deg));
        for k=1:g.order, check(isequal(sort(g.perms(k,:)),1:N),'perms row %d',k); end
        % mul/perm consistency + Latin square (sample for speed at N=36)
        for a2=1:g.order
            pa=g.perms(a2,:);
            check(isequal(sort(double(g.mul(a2,:))),1:g.order),'mul row %d',a2);
            for b2=1:g.order
                check(isequal(g.perms(double(g.mul(a2,b2)),:), pa(g.perms(b2,:))),'mul/perm (%d,%d)',a2,b2);
            end
            check(g.mul(a2,g.inv(a2))==g.identity,'inv %d',a2);
        end
        % bond invariance under every element
        base=sortrows(bonds);
        for k=1:g.order
            pk=g.perms(k,:); mp=[pk(bonds(:,1))', pk(bonds(:,2))'];
            mp=sortrows([min(mp,[],2),max(mp,[],2)]);
            check(isequal(mp,base),'bond set not invariant under %d',k);
        end
        % MIN_IMAGE_IH vs brute force (s=1/2)
        check_min_image(g, N);
        fprintf('  all checks passed.\n');
    end
    fprintf('\nALL TESTS PASSED.\n');
end

function check(c,msg,varargin), if ~c, error(['test_kagome_spacegroup: ' msg],varargin{:}); end, end

function check_min_image(g,N)
    d_loc=2; s=0.5;
    if N<=12, states=(0:2^N-1)'; else, states=double(randi([0 1],1500,N))*(d_loc.^(0:N-1))'; end
    states=int64(states); rb=states;
    for k=1:g.order, rb=min(rb,apply_perm_to_state(g.perms(k,:),states,d_loc,N)); end
    [reps,gmin]=min_image_Ih(states,g,s);
    check(isequal(reps,rb),'min_image reps != brute');
    for k=1:g.order
        sel=(gmin==k); if ~any(sel), continue; end
        check(isequal(int64(apply_perm_to_state(g.perms(k,:),states(sel),d_loc,N)),reps(sel)),'gmin %d',k);
    end
end
