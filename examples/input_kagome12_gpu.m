% N=12 kagome torus (2,0), C_6v space group, GPU FTLM smoke test (kernel path).
geometry='kagome'; kag_a=2; kag_b=0;
s_val=0.5; J=1.0; R=8; M_lz=60; ed_thresh=0;
lookup_method='bitmap'; entries_storage='host'; checkpoint=false; B_gpu=0;
T_range=logspace(-1,1,40);
