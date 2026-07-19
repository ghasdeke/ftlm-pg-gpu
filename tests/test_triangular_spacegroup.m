function test_triangular_spacegroup()
%TEST_TRIANGULAR_SPACEGROUP  Verify the triangular torus space-group provider.
%   (a,b)=(2,2) [N=12] and (4,0) [N=16]: order (multiple of ncells, <=12*ncells),
%   NN bonds (3N, degree 6), group axioms (Latin-square mul, inverses), bond
%   invariance under every element, and MIN_IMAGE_IH consistency vs brute force.
%
%   The order check is SOFT (order = ncells * faithful-point-group-order,
%   <= 12*ncells) because the single C_6v-site action can be unfaithful on a
%   small torus; the structural checks below MUST hold for any valid quotient.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    rng(7);
    cases = [2 2; 4 0];                 % N=12, N=16
    for ci = 1:size(cases,1)
        a=cases(ci,1); b=cases(ci,2);
        [g, bonds] = triangular_spacegroup(a,b);
        N=g.N; fprintf('=== triangular (a,b)=(%d,%d): N=%d, |G|=%d (%s), %d bonds ===\n', ...
            a,b,N,g.order,g.point_group,size(bonds,1));
        check(mod(g.order, g.ncells)==0 && g.order <= 12*g.ncells, ...
            'order %d not in {k*ncells : k<=12}', g.order);
        check(size(bonds,1) == 3*N, 'bond count %d != 3N=%d', size(bonds,1), 3*N);
        deg = accumarray([bonds(:,1);bonds(:,2)],1,[N 1]);
        check(all(deg==6), 'every site degree 6 (got min %d max %d)', min(deg), max(deg));
        for k=1:g.order, check(isequal(sort(g.perms(k,:)),1:N),'perms row %d',k); end
        % mul/perm consistency + Latin square
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

function check(c,msg,varargin), if ~c, error(['test_triangular_spacegroup: ' msg],varargin{:}); end, end

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
