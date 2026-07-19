function [IH_p, Irrep_Ag, Irrep_Au, Irrep_T1g, Irrep_T1u,...
Irrep_T2g, Irrep_T2u, Irrep_Fg, Irrep_Fu,...
Irrep_Hg, Irrep_Hu] = ...
icosahedron_ih_representation_matrices_and_irreps(x)

% C5-Generator

cyc5 = zeros(5,2);

cyc5(:,1) = [2,10,9,5,12];
cyc5(:,2) = [3,6,11,8,4];

% C3-Generator

cyc3 = zeros(3,4);

cyc3(:,1)  = [1,12,2];
cyc3(:,2)  = [7,11,8];
cyc3(:,3)  = [4,6,9];
cyc3(:,4)  = [3,10,5];


% Ci-Generator

cyc_ci = zeros(2,6);

cyc_ci(:,1) = [1,7];
cyc_ci(:,2) = [2,8];
cyc_ci(:,3) = [3,9];
cyc_ci(:,4) = [4,10];
cyc_ci(:,5) = [5,6];
cyc_ci(:,6) = [11,12];


% Aufbau einer C5-Darstellungsmatrix

C5 = zeros(12);
C5(1,1) = 1;
C5(7,7) = 1;

for m = 1 : 2

	for n = 1 : 5

		if n < 5
			aux = n + 1;
		else
			aux = 1;
		end

		a = cyc5(aux,m);
		b = cyc5(n,m);

		C5(a,b) = 1;
	
	end
end


% Aufbau einer C3-Darstellungsmatrix

C3 = zeros(12);

for m = 1 : 4

	for n = 1 : 3

		if n < 3
			aux = n + 1;
		else
			aux = 1;
		end

		a = cyc3(aux,m);
		b = cyc3(n,m);

		C3(a,b) = 1;
	
	end
end


% Aufbau einer Ci-Darstellungsmatrix

Ci = zeros(12);

for m = 1 : 6

	for n = 1 : 2

		if n < 2
			aux = n + 1;
		else
			aux = 1;
		end

		a = cyc_ci(aux,m);
		b = cyc_ci(n,m);

		Ci(a,b) = 1;
	
	end
end


%%%% neu ist hier, dass ich dem C5-Generator
%%%% und dem C3-Generator jetzt auch die Matrizen
%%%% für alle Irreps der Gruppe I zuordne.

%%%% Die Irreps sind A, T1, T2, F und H und die Generatormatrizen
%%%% für C5 und C3 sind auf S. 655 von Altmann und Herzig gegeben.
%%%% Ich brauche hier nur diese beiden Generatoren.
%%%% Die Abbildungen auf S. 644 von Altmann und Herzig legen nahe,
%%%% dass der C3-Generator so gewählt wird, dass ein Dreieck, durch
%%%% das die C3-Achse läuftan einer Ecke ein Zentrum hat, durch
%%%% das der C5-Generator verläuft (das ist hier Spin 1,
%%%%% die Generatoren C5 und C3, die über die Zyklen oben
%%%%% definiert sind, sollten also korrekt sein). 

Irrep_A = zeros(60,1);
Irrep_T1 = zeros(3,3,60);
Irrep_T2 = zeros(3,3,60);
Irrep_F = zeros(4,4,60);
Irrep_H = zeros(5,5,60);

%% definiere jetzt die notwendigen Größen, die unten auf S. 655 gegeben sind

g_p = (sqrt(5)+1)/2;
g_m = (sqrt(5)-1)/2;
t = sqrt(5);
l = exp((1i)*atan(sqrt(5/3))); % lambda
om = exp(2*pi*(1i)/3); % omega



Irrep_A(1) = 1;

Irrep_T1(:,:,1) = ...
	   1/2 * [g_m, (-1i), (1i) * (-g_p)
		  (-1i), g_p, (-1) * g_m
		  (-1i) * g_p, (-1)*g_m, 1];

Irrep_T2(:,:,1) = ...
	   1/2 * [(-1) * g_p, (-1i), (1i) * g_m
		  (-1i), (-1)*g_m, g_p
		  (1i) * g_m, g_p, 1];

Irrep_F(:,:,1) = ...
	   1/4 * [(-1), (-1)*t, (-1i) * t, (-1i) * t
		  (-1)*t, (-1), 3*(1i), (-1i)
		(-1i)*t, 3*(1i), 1, 1
		(-1i)*t, (-1i), 1, (-3)];

Irrep_H(:,:,1) = ...
	   1/2 * [0, l*l*conj(om), (-l), (1i)*l*(-conj(om)), (1i)*l*(-om)
		 (conj(l))^2 * om, 0, (-conj(l)), (1i) * conj(l)*(-om), (1i)*conj(l)*(-conj(om))
		(-conj(l)), (-l), 1, 0,(1i),
		(1i)*conj(l)*(-om), (1i)*l*(-conj(om)), 0, (-1), (-1)
		(1i)*conj(l)*(-conj(om)), (1i)*l*(-om), (1i), (-1), 0];

%%%%%% Irrep-Matrizen für C5-Generator berechnet

Irrep_A(2) = 1;

Irrep_T1(:,:,2) = ...
		[0, 0, (-1i)
		(-1i), 0, 0
		  0,  (-1), 0];

Irrep_T2(:,:,2) = ...
		[0, 0, (-1i)
		(-1i), 0, 0
		  0,  (-1), 0];

Irrep_F(:,:,2) = ...
		[1, 0, 0,  0
		 0, 0, 0, (-1i)
		 0, (-1i), 0, 0,
		 0,   0,  (-1), 0];

Irrep_H(:,:,2) = ...
		[om, 0, 0, 0, 0
		  0, conj(om), 0, 0, 0
		  0, 0, 0, 0, (-1i)
		  0, 0, (-1i), 0, 0
		  0, 0,  0, (-1), 0];


%%%%%% Irrep-Matrizen für C3-Generator berechnet



M(:,:,1) = C5;
M(:,:,2) = C3;
%M(:,:,3) = Ci;

dim = size(M,3);
treffer = 1; %Initialisierung; Wert von "treffer" ist egal, Hauptsache es ist zu Beginn größer als null und kleiner als zwei


while treffer > 0
counter = 0;
    
    m = 0;
	while (counter == 0 && m < dim )
        m = m + 1;
        n = 0;
        while (counter == 0 && n < dim )
            n = n + 1;

			N = M(:,:,m)*M(:,:,n);

		    Irrep_A_aux = Irrep_A(m)*Irrep_A(n);
		    Irrep_T1_aux = Irrep_T1(:,:,m)*Irrep_T1(:,:,n);
		    Irrep_T2_aux = Irrep_T2(:,:,m)*Irrep_T2(:,:,n);
		    Irrep_F_aux = Irrep_F(:,:,m)*Irrep_F(:,:,n);
		    Irrep_H_aux = Irrep_H(:,:,m)*Irrep_H(:,:,n);


            tester = 0;
            for l = 1 : dim
                K = M(:,:,l);
                tester = tester + isequal(N,K); %überprüfen, ob die Matrix N neu ist
            end

                if tester == 0
                    
                    counter = counter + 1;
                    treffer = counter;
                    M(:,:,dim+1) = N;

               	Irrep_A(dim+1) = Irrep_A_aux;
               	Irrep_T1(:,:,dim+1) = Irrep_T1_aux;
               	Irrep_T2(:,:,dim+1) = Irrep_T2_aux;
               	Irrep_F(:,:,dim+1) = Irrep_F_aux;
               	Irrep_H(:,:,dim+1) = Irrep_H_aux;
                    
                end %Ende der if-Bedingung
			
        end%Ende der while-Schleife über n
	end %Ende der while-Schleife über m

	dim = dim + treffer;
	
    if counter == 0
        treffer = 0;
    end
    

end %Ende der while-Schleife

IH_p = zeros(12,12,120);

for m = 1 : 60

	IH_p(:,:,(2*m-1)) = M(:,:,m);

	IH_p(:,:,2*m) = M(:,:,m) * Ci;


end

Irrep_Ag = ones(120,1);

Irrep_T1g = zeros(3,3,120);
Irrep_T1u = zeros(3,3,120);
Irrep_T2g = zeros(3,3,120);
Irrep_T2u = zeros(3,3,120);
Irrep_Fg = zeros(4,4,120);
Irrep_Fu = zeros(4,4,120);
Irrep_Hg = zeros(5,5,120);
Irrep_Hu = zeros(5,5,120);


for m = 1 : 60

	Irrep_Au(2*m-1) = 1;
	Irrep_Au(2*m) =  -1;

%%%
	Irrep_T1g(:,:,2*m-1) =  Irrep_T1(:,:,m);
	Irrep_T1u(:,:,2*m-1) =  Irrep_T1(:,:,m);

	Irrep_T1g(:,:,2*m) =  Irrep_T1(:,:,m);
	Irrep_T1u(:,:,2*m) =  (-1) * Irrep_T1(:,:,m);

%%%
	Irrep_T2g(:,:,2*m-1) =  Irrep_T2(:,:,m);
	Irrep_T2u(:,:,2*m-1) =  Irrep_T2(:,:,m);

	Irrep_T2g(:,:,2*m) =  Irrep_T2(:,:,m);
	Irrep_T2u(:,:,2*m) =  (-1) * Irrep_T2(:,:,m);

%%%

	Irrep_Fg(:,:,2*m-1) =  Irrep_F(:,:,m);
	Irrep_Fu(:,:,2*m-1) =  Irrep_F(:,:,m);

	Irrep_Fg(:,:,2*m) =  Irrep_F(:,:,m);
	Irrep_Fu(:,:,2*m) =  (-1) * Irrep_F(:,:,m);

%%%

	Irrep_Hg(:,:,2*m-1) =  Irrep_H(:,:,m);
	Irrep_Hu(:,:,2*m-1) =  Irrep_H(:,:,m);

	Irrep_Hg(:,:,2*m) =  Irrep_H(:,:,m);
	Irrep_Hu(:,:,2*m) =  (-1) * Irrep_H(:,:,m);

end











