import numpy as np
from soint2dcfun import *

def sint2d(din,dip,mask,niter=100,eps=0.01,ns=1,order=1,verb=1):
	'''
	
	sint2d: Interpolation for sparse data in 2D, e.g., well logs

	Yangkang Chen
	Aug, 11, 2021
	Ported to Python in Jun, 15, 2023
	
	INPUT
	din:
	dip:
	mask:
	eps:
	ns:
	order:
	
	OUTPUT
	dout
	
	EXAMPLE
	
	
	'''
	from .solvers import conjgrad
	from .pwsmooth import pwsmooth_set,pwsmooth_lop
	from .operators import mask_lop
	
	[n1,n2]=din.shape;
	n12=n1*n2;
	mm=din.flatten(order='F'); #din is a 2D matrix
	mask=mask.flatten(order='F');

	#figure out scaling and make known data mask
	lam=0;
	for ii in range(n12):
		if mask[ii] !=0:
			lam=lam+1;
	lam=np.sqrt(lam/n12);
	
	w1 = pwsmooth_set(dip,n1,n2,ns,order,eps);
	par_L={'nm':n12,'nd':n12,'mask':mask}
	par_S={'nm':n12,'nd':n12,'dip':dip,'w1':w1,'ns':ns,'order':order,'eps':eps}
	
	eps_cg=lam*lam;
	tol_cg=0.0000019209;
	ifhasp0=1;
	
	# Conjugate Gradient with Shaping
	mm = conjgrad(None, mask_lop, pwsmooth_lop, mm, mm, mm, eps_cg, tol_cg, niter,ifhasp0,[],par_L,par_S,verb);
	dout=mm.reshape(n1,n2,order='F')
	
	return dout
	
	
	
def sint2dc(din,dip,mask,niter=100,eps=0.01,ns=1,order=1,verb=1):
	'''
	
	sint2dc: Interpolation for sparse data in 2D, e.g., well logs

	Yangkang Chen
	Aug, 11, 2021
	Ported to Python in Jun, 15, 2023
	
	INPUT
	din:
	dip:
	mask:
	eps:
	ns:
	order:
	
	OUTPUT
	dout
	
	EXAMPLE
	
	
	'''
	from .solvers import conjgrad
	from .pwsmooth import pwsmooth_set,pwsmooth_lop
	from .operators import mask_lop
	
	[n1,n2]=din.shape;
# 	n12=n1*n2;
# 	mm=din.flatten(order='F'); #din is a 2D matrix
# 	mask=mask.flatten(order='F');
# 	print('Check 0')
# 	#figure out scaling and make known data mask
# 	lam=0;
# 	for ii in range(n12):
# 		if mask[ii] !=0:
# 			lam=lam+1;
# 	lam=np.sqrt(lam/n12);
# 	
# 	w1 = pwsmooth_set(dip,n1,n2,ns,order,eps);
# 	print('Check 1')
# 	par_L={'nm':n12,'nd':n12,'mask':mask}
# 	par_S={'nm':n12,'nd':n12,'dip':dip,'w1':w1,'ns':ns,'order':order,'eps':eps}
# 	
# 	eps_cg=lam*lam;
# 	tol_cg=0.0000019209;
# 	ifhasp0=1;
# 	
# 	print('Check 2')
# 	# Conjugate Gradient with Shaping
# 	[mm ] = conjgrad(None, mask_lop, pwsmooth_lop, mm, mm, mm, eps_cg, tol_cg, niter,ifhasp0,[],par_L,par_S,verb);
# 	dout=mm.reshape(mm,n1,n2,order='F')
# 	
# 	dout=csint2d(din,dip,mask,)
# 	
	din=np.float32(din).flatten(order='F');
	mask=np.float32(mask).flatten(order='F');
	dip=np.float32(dip).flatten(order='F');
	
	dout=csint2d(din,dip,mask,n1,n2,niter,ns,order,verb,eps);
	dout=dout.reshape(n1,n2,order='F');
	
	return dout	
	
	
