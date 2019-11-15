***,emmim + h electric field
memory,100,m

basis
set,orbital;	!for orbital
default=avdz
set,jkfit;		!for JK integrals
default=avdz/jkfit
set,mp2fit;		!for E2disk/E2exch-disp
default=avdz/mp2fit
set,dflhf;		!for LHF
default=avdz/jkfit
end

nosym;noorient;angstrom;
geometry={
C 1.1530248 0.52929646 -0.12557359
N 2.3766766 -0.015184514 0.1024829
C 2.2873282 -1.3947171 0.004785758
C 0.9812654 -1.6944653 -0.2901313
N 0.2965558 -0.49395415 -0.37167448
C -1.1526698 -0.36783463 -0.62572074
C -1.9939742 -0.5024993 0.6406791
H -1.8431337 -1.4864252 1.1268218
H -1.7619543 0.29085904 1.3788607
H -1.3177977 0.60684305 -1.1242282
H -1.4106971 -1.1502981 -1.3662536
H 0.49002415 -2.6590042 -0.4524029
H 3.1550078 -2.0462186 0.14799531
C 3.6052637 0.72865 0.39107847
H 3.4873402 1.3310506 1.3120092
H 4.42416 0.0048906235 0.54579234
H 3.8693109 1.3883828 -0.45734793
C 0.8165846 1.9769865 -0.13559024
H 0.595259 2.330751 -1.1650416
H -0.071480446 2.1885266 0.49225134
H 1.655723 2.5809536 0.25260288
H -3.066119 -0.41652036 0.37714565
H 0.47541565 2.7699509 -3.5214667
}

cb= 2100.2

	shift_mim=.068574661 !shift for bulk xc potential

	dfit,basis_coul=jkfit,basis_exch=jkfit

	!monomer B
	dummy,23
	charge=0
	{df-ks,pbe; start,atdens; asymp,shift_mim; save,$cb}

Property 
density,$cb
EF,23