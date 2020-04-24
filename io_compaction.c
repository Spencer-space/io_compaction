#include "io_compaction.h"
#include <errno.h>
#include <sys/types.h>
#include <pwd.h>
#include <sys/stat.h>

/* Preamble - Setup
   X - Solution vector filled when solving for the next timestep
   R - Residual vector for timestepping
   Xold - Solution vector for the current time when solution is known
   
   Using a staggered grid, velocity is positioned between nodes, with the last velocity point in the vector being redundant
*/

PetscClassId classid[8];

/*-----------------------------------------------------------------------*/
int main(int argc,char **argv)
/*-----------------------------------------------------------------------*/
{
  AppCtx         user;
  PetscErrorCode ierr;
  
  /* always initialize PETSc first */
  PetscInitialize(&argc,&argv,(char *)0,PETSC_NULL);
  user.comm = PETSC_COMM_WORLD;
  
  /* set hardwired options */
  PetscOptionsSetValue(NULL,"-ksp_gmres_restart","300");
  //PetscOptionsSetValue(NULL,"-ksp_monitor","");
  //PetscOptionsSetValue(NULL,"-snes_monitor","");
  PetscOptionsSetValue(NULL,"-P_snes_linesearch_type","basic");
  PetscOptionsSetValue(NULL,"-H_snes_linesearch_type","basic");
  PetscOptionsSetValue(NULL,"-H_pc_type","lu");
  PetscOptionsSetValue(NULL,"-H_snes_stol","1.0e-8");
  PetscOptionsSetValue(NULL,"-H_snes_rtol","1.0e-6");
  PetscOptionsSetValue(NULL,"-H_snes_abstol","1.0e-7");
  PetscOptionsSetValue(NULL,"-snes_converged_reason","");
  PetscOptionsInsert(NULL,&argc,&argv,PETSC_NULL);
  
  /* set up parameter structure */
  ierr = PetscBagCreate(user.comm,sizeof(Parameter),&(user.bag));  CHKERRQ(ierr);
  ierr = PetscBagGetData(user.bag,(void**)&user.param);  CHKERRQ(ierr);
  ierr = PetscBagSetName(user.bag,"par","parameters for compaction problem");  CHKERRQ(ierr);
  ierr = ParameterSetup(user);CHKERRQ(ierr);
  
  /* Create output directory if it doesn't already exist */
  ierr = projCreateDirectory("outputs"); CHKERRQ(ierr);
  ierr = projCreateDirectory("SS_outputs"); CHKERRQ(ierr);
  /* Set up solution vectors and SNES */
  ierr = DMDASNESsetup(&user);CHKERRQ(ierr);
  
  /* if a restart file is specified, load it in, else set-up with initialisation */
  if (user.param->restart_step != 0) {
    ierr = RestartFromFile(&user);CHKERRQ(ierr);
  } else {
    /* Initialise vectors and parameters that need calculating */
    ierr = Initialisation(&user,user.XP,user.XH,user.XPipe,user.Haux,user.Paux);CHKERRQ(ierr);
  }

  /* Main timestepping function */
  ierr = TimeStepping(&user);CHKERRQ(ierr);
  
  /* Clean-up by deleting vectors etc */
  ierr = CleanUp(&user);CHKERRQ(ierr);
  
  return 0;
}

/* ------------------------------------------------------------------- */
PetscErrorCode Initialisation(AppCtx *user, Vec XP, Vec XH, Vec XPipe, Vec Haux, Vec Paux)
/* ------------------------------------------------------------------- */
{
  Parameter       *p = user->param;
  PetscErrorCode  ierr;
  FieldP          *xp;
  FieldH          *xh;
  FieldPipe       *xpipe;
  AuxFieldP       *paux;
  AuxFieldH       *haux;
  PetscInt        i;
  PetscReal       dr;
  PetscFunctionBegin;
  
  /* set non-dim base */
  p->base = p->R_cmb/p->R;
  
  /* Spatial grid spacing (constant) */
  dr = (1-p->base)/(p->ni-2);
  
  /* Reference values and non-dim values */
  p->psi_0 = p->Psi_0/(4.0/3.0 * PETSC_PI * (pow(p->R,3)-pow(p->R_cmb,3))); // set reference constant heating rate in W/m3
  p->Pe = p->psi_0*p->R*p->R/(p->L*p->kappa*p->rho_0); // Peclet number
  p->St = p->L/(p->ce*p->T_0); // Stefan number
  p->phi_0 = pow(p->psi_0*p->R*p->eta_l/(p->L*p->rho_0*p->K_0*p->del_rho*p->g),1/p->perm); // reference porosity
  p->zeta_0 = p->eta/p->phi_0;
  p->delta = p->zeta_0*p->K_0*pow(p->phi_0,p->perm)/(p->eta_l*p->R*p->R);
  p->P_0 = p->zeta_0*p->psi_0/(p->rho_0*p->L); // reference pressure
  
  /* Get the initial (empty) solution vectors */
  ierr = DMDAVecGetArray(user->daP,XP,&xp);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daH,XH,&xh);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daPipe,XPipe,&xpipe);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daPaux,Paux,&paux);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daHaux,Haux,&haux);CHKERRQ(ierr);
  
  /* if not initialising from some other state, set up one here */
  for (i=0; i<p->ni; i++) {
    paux[i].r  = p->base + dr*(i-0.5); // set up position vector (cell centers)
    if (paux[i].r < 0.9) {
      xh[i].H = 1.0;
    } else {
      xh[i].H = 1.0*(1-paux[i].r)/(1-0.9); // set up enthalpy
    }
  }
  
  ierr = PAuxParamsCalc(user,xp,xpipe,haux,paux);CHKERRQ(ierr);
  
  ierr = DMDAVecRestoreArray(user->daP,XP,&xp);CHKERRQ(ierr); // restore vector
  ierr = DMDAVecRestoreArray(user->daH,XH,&xh);CHKERRQ(ierr); // restore vector
  ierr = DMDAVecRestoreArray(user->daPipe,XPipe,&xpipe);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daPaux,Paux,&paux);CHKERRQ(ierr); // restore vector
  ierr = DMDAVecRestoreArray(user->daHaux,Haux,&haux);CHKERRQ(ierr); // restore vector
  
  ierr = VecCopy(user->XP,user->XPtheta);CHKERRQ(ierr); // P theta vector initially takes the setup state
  ierr = VecCopy(user->Paux,user->Pauxtheta);CHKERRQ(ierr); // P theta vector initially takes the setup state
  
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode TimeStepping(AppCtx *user)
/* ------------------------------------------------------------------- */
{
  Parameter    *p = user->param;
  PetscBool   STEADY=PETSC_FALSE, converged;
  PetscErrorCode  ierr;
  PetscReal   dt = p->dt;
  PetscReal   Pc_original;
  PetscReal   eps=1e-7, RH2, RP2, RPipe2;
  PetscInt    SNESit;
  PetscFunctionBegin;
  
  /* Output initial state */
  ierr = DoOutput(user,p->it,STEADY);CHKERRQ(ierr);
  
  /* If big pc used, set it small to be gradually increased */
  Pc_original = p->Pc;
  if (p->Pc > 0.5) {
    p->Pc = 0.5;
  }
  
  while (p->t <= p->tmax && p->it <= p->ns) {
    printf("########## step %d ################ initial timestep %+1.4e \n",p->it,p->dt);
    ierr = VecCopy(user->XP,user->XPold);CHKERRQ(ierr); // copy current timestep's solution and stores it as Xold.
    ierr = VecCopy(user->XH,user->XHold);CHKERRQ(ierr); // copy current timestep's solution and stores it as Xold.
    ierr = VecCopy(user->XPipe,user->XPipeold);CHKERRQ(ierr); // copy current timestep's solution and stores it as Xold.
    ierr = VecCopy(user->Haux,user->Hauxold);CHKERRQ(ierr);
    ierr = VecCopy(user->Paux,user->Pauxold);CHKERRQ(ierr);
    
    converged = PETSC_FALSE;
    FLAGH:
    printf("   >>>>>>>>>>>>>>>>>>>>[step %d] timestep %+1.4e \n",p->it,p->dt);
    if (p->dt < 1e-12) {
      ierr = CHANGE_CONSOLE_COLOUR(user->comm,BOLD_RED);CHKERRQ(ierr);
      ierr = PetscPrintf(user->comm,"TIMESTEP REDUCED BELOW TOLERANCE at it %i \n",p->it);CHKERRQ(ierr);
      ierr = CHANGE_CONSOLE_COLOUR(user->comm,COLOUR_RESET);CHKERRQ(ierr);
      break;
     }
     
    for(SNESit=0; SNESit<p->SNESit; SNESit++) {
      
      /* Do enthalpy solve */
      ierr = SNESSolve(user->snesH,PETSC_NULL,user->XH);CHKERRQ(ierr); // non-linear solver
      ierr = SNESGetConvergedReason(user->snesH,&user->reasonH);CHKERRQ(ierr);
      ierr = SNESComputeFunction(user->snesH,user->XH,user->RH);CHKERRQ(ierr); // make sure I have the correct auxilliary parameters

      /* Now do pressure solve */
      ierr = SNESSolve(user->snesP,PETSC_NULL,user->XP);CHKERRQ(ierr); // non-linear solver
      ierr = SNESGetConvergedReason(user->snesP,&user->reasonP);CHKERRQ(ierr);
      ierr = SNESComputeFunction(user->snesP,user->XP,user->RP);CHKERRQ(ierr); // make sure I have the correct auxilliary parameters
      
      /* Now do pipe solve */
      ierr = SNESSolve(user->snesPipe,PETSC_NULL,user->XPipe);CHKERRQ(ierr); // non-linear solver
      ierr = SNESGetConvergedReason(user->snesPipe,&user->reasonPipe);CHKERRQ(ierr);
      ierr = SNESComputeFunction(user->snesPipe,user->XPipe,user->RPipe);CHKERRQ(ierr); // make sure I have the correct auxilliary parameters

      ierr = SNESComputeFunction(user->snesH,user->XH,user->RH);CHKERRQ(ierr);
      ierr = SNESComputeFunction(user->snesP,user->XP,user->RP);CHKERRQ(ierr);
      /* no need to compute pipe residual as it was evaluated above after its SNESSolve */
      ierr = VecNorm(user->RH,NORM_2,&RH2);CHKERRQ(ierr);
      ierr = VecNorm(user->RP,NORM_2,&RP2);CHKERRQ(ierr);
      ierr = VecNorm(user->RPipe,NORM_2,&RPipe2);CHKERRQ(ierr);
      
      printf("[SNESit %d] reasons %d %d %d : norms %+1.4e  %+1.4e  %+1.4e\n",SNESit,user->reasonH,user->reasonP,user->reasonPipe,RH2,RP2,RPipe2);
      
      /*
       <DAM>
       Reduce dt less aggresively, 2x instead of 10x
      */
      if (RH2 < eps && RP2 < eps && RPipe2 < eps) {
        converged = PETSC_TRUE;
        break;
      }
      if (user->reasonH < 0) { p->dt = p->dt/2.0; goto FLAGH; } // if diverged from max-iterations, go back and try with smaller timestep
      if (user->reasonP < 0) { p->dt = p->dt/2.0; goto FLAGH; } // if diverged from max-iterations, go back and try with smaller timestep
      if (user->reasonPipe < 0) { p->dt = p->dt/2.0; goto FLAGH; } // if diverged from max-iterations, go back and try with smaller timestep
      /* if timestep is too small, exit, it's not doing anything */
      if (p->dt < 1e-12) {
        break;
      }
    }
    if (!converged && p->dt > 1e-12) {
      p->dt = p->dt/2.0; goto FLAGH;
    } else if (!converged) {
      ierr = CHANGE_CONSOLE_COLOUR(user->comm,BOLD_RED);CHKERRQ(ierr);
      ierr = PetscPrintf(user->comm,"TIMESTEP REDUCED BELOW TOLERANCE at it %i \n",p->it);CHKERRQ(ierr);
      ierr = CHANGE_CONSOLE_COLOUR(user->comm,COLOUR_RESET);CHKERRQ(ierr);
      goto FLAGERR;
    }
    
    p->t = p->t + p->dt; // update time
    
    /*  Additional files to check outputs through run  */
    if ((p->out_freq != 0) && (p->it % p->out_freq == 0)) { ierr = DoOutput(user,p->it,STEADY); CHKERRQ(ierr); }
    if (p->it % 10 == 0 || p->dt != dt) {
      ierr = PetscPrintf(user->comm,"---------- Timestep - %i, time - %e -------- dt = %e ---\n",p->it,p->t,p->dt);CHKERRQ(ierr);
    }
    
    /* Every 100 timesteps check if steady state has been achieved, if so break */
    if(p->it % 100 == 0) {
      STEADY = SteadyStateChecker(user);
      if (STEADY && p->Pc == Pc_original && p->D == 1e-8) { // && p->D == D_original
        ierr = CHANGE_CONSOLE_COLOUR(user->comm,BOLD_GREEN);CHKERRQ(ierr);
        ierr = PetscPrintf(user->comm,"STEADY STATE ACHIEVED AFTER %i TIMESTEPS \n",p->it);CHKERRQ(ierr);
        ierr = CHANGE_CONSOLE_COLOUR(user->comm,COLOUR_RESET);CHKERRQ(ierr);
        break;
      }
      /* if not using full desired Pc, increase */
      if (STEADY && p->Pc != Pc_original) {
        p->Pc = p->Pc + 0.2;
      }
      /* check that Pc hasn't been overshot */
      if (p->Pc > Pc_original) {
        p->Pc = Pc_original;
      }
    }
    p->it++;
    

    /*
     <DAM>
     Gradually increase dt (5%), but only if step is a multiple of 10.
     The value of 10 is arbitrary, we just want a rule to slowly grow dt.
     Always keep dt less than the user specified time step.
     We could get rid of p->it%10 and instead simply grow dt by a factor 1.005.
     That might be nicer and easier to understand.
    */
    if (p->restart_step == 0 && p->dt < dt && p->it%10 == 0) {
      p->dt = PetscMin( dt , 1.05 * p->dt );
    }
    if (p->restart_step != 0 && p->dt != 5e-4) {
      p->dt = 1.05*p->dt; // if restarting, dt is set to what it was when restarting so just let it grow.
    }
    if (p->dt > 5e-4) {
      p->dt = 5e-4;
    }
    
  }
  
  FLAGERR:ierr = DoOutput(user,p->it,STEADY);CHKERRQ(ierr);
  ierr = PetscPrintf(user->comm,"Computation stopped after %d timesteps, at time %e \n",p->it,p->t);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode FormResidualH(SNES snesH, Vec XH, Vec RH, void *ptr)
/* ------------------------------------------------------------------- */
{
  AppCtx          *user = (AppCtx*)ptr;
  Parameter       *p = user->param;
  PetscErrorCode  ierr;
  PetscInt        i, is, ie;
  const FieldH    *xh;
  FieldH          *rh, *xhold, *xhtheta;
  FieldP          *xptheta;
  FieldPipe       *xpipetheta;
  AuxFieldH       *haux, *hauxtheta;
  AuxFieldP       *pauxtheta;
  PetscFunctionBegin;
  
  /* get the residual array */
  ierr = DMDAVecGetArray(user->daH,RH,&rh);CHKERRQ(ierr);
  
  /* get the current guess of solution */
  ierr = DMDAVecGetArrayRead(user->daH,XH,&xh);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daHaux,user->Haux,&haux);CHKERRQ(ierr);
  
  /* calculate auxilliary parameters associated with this solver */
  ierr = HAuxParamsCalc(user,xh,haux);CHKERRQ(ierr);
  
  /* theta method for XH */
  ierr = DMDAVecRestoreArrayRead(user->daH,XH,&xh);CHKERRQ(ierr);
  ierr = VecCopy(XH,user->XHtheta);CHKERRQ(ierr);
  ierr = VecAXPBY(user->XHtheta,1.0-p->theta,p->theta,user->XHold);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(user->daH,XH,&xh);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daH,user->XHtheta,&xhtheta);CHKERRQ(ierr);
  /* theta method for Haux*/
  ierr = DMDAVecRestoreArray(user->daHaux,user->Haux,&haux);CHKERRQ(ierr);
  ierr = VecCopy(user->Haux,user->Hauxtheta);CHKERRQ(ierr);
  ierr = VecAXPBY(user->Hauxtheta,1.0-p->theta,p->theta,user->Hauxold);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daHaux,user->Hauxtheta,&hauxtheta);CHKERRQ(ierr);
  /* theta method for XP */
  ierr = VecCopy(user->XP,user->XPtheta);CHKERRQ(ierr);
  ierr = VecAXPBY(user->XPtheta,1.0-p->theta,p->theta,user->XPold);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daP,user->XPtheta,&xptheta);CHKERRQ(ierr);
  /* theta method for Paux */
  ierr = VecCopy(user->Paux,user->Pauxtheta);CHKERRQ(ierr);
  ierr = VecAXPBY(user->Pauxtheta,1.0-p->theta,p->theta,user->Pauxold);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daPaux,user->Pauxtheta,&pauxtheta);CHKERRQ(ierr);
  /* theta method for pipe */
  ierr = VecCopy(user->XPipe,user->XPipetheta);CHKERRQ(ierr);
  ierr = VecAXPBY(user->XPipetheta,1.0-p->theta,p->theta,user->XPipeold);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daPipe,user->XPipetheta,&xpipetheta);CHKERRQ(ierr);
  
  /* get the solution from last timestep */
  ierr = DMDAVecGetArray(user->daH,user->XHold,&xhold);CHKERRQ(ierr);
  
  /* get dimensions of LOCAL piece of grid (in parallel, this is not full grid) */
  ierr = DMDAGetGridInfo(user->daH,&is,0,0,&ie,0,0,0,0,0,0);CHKERRQ(ierr);
  
  /* apply boundary condition at r=0 */
  if (is==0) {
     i = 0; is++;
     rh[i].H  = xh[i].H - xh[i+1].H;
  }
  
  /* apply boundary condition at r=r_max */
  if (ie==p->ni) {
     i = p->ni-1; ie--;
     rh[i].H  = 0.5*(xh[i].H + xh[i-1].H) - p->H_end;
  }
  
  /* interior of the domain */
  for (i=is; i<ie; i++) {
    rh[i].H  = EnthalpyResidual(user,xh,xhold,xhtheta,xptheta,xpipetheta,hauxtheta,pauxtheta,i);
  }
  
  /* clean up */
  ierr = DMDAVecRestoreArrayRead(user->daH,XH,&xh);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daH,user->XHold,&xhold);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daH,user->XHtheta,&xhtheta);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daP,user->XPtheta,&xptheta);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daPipe,user->XPipetheta,&xpipetheta);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daHaux,user->Hauxtheta,&hauxtheta);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daPaux,user->Pauxtheta,&pauxtheta);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daH,RH,&rh);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode FormResidualP(SNES snesP, Vec XP, Vec RP, void *ptr)
/* ------------------------------------------------------------------- */
{
  AppCtx          *user = (AppCtx*)ptr;
  Parameter       *p = user->param;
  PetscReal       dr = (1 - p->base)/(p->ni-2);
  PetscErrorCode  ierr;
  PetscInt        i, is, ie;
  const FieldP    *xp;
  FieldP          *rp;
  FieldPipe       *xpipe;
  AuxFieldH       *haux;
  AuxFieldP       *paux;
  PetscFunctionBegin;
  
  /* get the residual array */
  ierr = DMDAVecGetArray(user->daP,RP,&rp);CHKERRQ(ierr);
  
  /* get the current guess of solution */
  ierr = DMDAVecGetArrayRead(user->daP,XP,&xp);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daPaux,user->Paux,&paux);CHKERRQ(ierr);
  
  /* get vector of solution from other SNESs */
  ierr = DMDAVecGetArray(user->daPipe,user->XPipe,&xpipe);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daHaux,user->Haux,&haux);CHKERRQ(ierr);
  
  /* get dimensions of LOCAL piece of grid (in parallel, this is not full grid) */
  ierr = DMDAGetGridInfo(user->daP,&is,0,0,&ie,0,0,0,0,0,0);CHKERRQ(ierr);
  
  /* calculate auxilliary parameters each Newton step from solution guess */
  ierr = PAuxParamsCalc(user,xp,xpipe,haux,paux);CHKERRQ(ierr);
  
  /* apply boundary condition at r=0 */
  if (is==0) {
     i = 0; is++;
     if (0.5*(haux[i].phi+haux[i+1].phi) > 1e-8) {
       rp[i].P  = xp[i].P - xp[i+1].P + dr*(1-p->phi_0*0.5*(haux[i].phi+haux[i+1].phi))/p->delta; // from q = 0 at base
     } else {
       rp[i].P = 0.5*(xp[i].P + xp[i+1].P);
     }
  }
  
  /* apply boundary condition at r=r_max */
  if (ie==p->ni) {
     i = p->ni-1; ie--;
     rp[i].P  = 0.5*(xp[i].P + xp[i-1].P); // no liquid overpressure at surface
  }
  
  /* interior of the domain */
  for (i=is; i<p->ni-1; i++) {
    rp[i].P  = PressureResidual(user,xp,xpipe,haux,paux,i);
  }
  
  /* clean up */
  ierr = DMDAVecRestoreArrayRead(user->daP,XP,&xp);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daPipe,user->XPipe,&xpipe);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daHaux,user->Haux,&haux);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daPaux,user->Paux,&paux);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daP,RP,&rp);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode FormResidualPipe(SNES snesPipe, Vec XPipe, Vec RPipe, void *ptr)
/* ------------------------------------------------------------------- */
{
  AppCtx          *user = (AppCtx*)ptr;
  Parameter       *p = user->param;
  PetscErrorCode  ierr;
  PetscInt        i, is, ie;
  const FieldPipe *xpipe;
  FieldPipe       *rpipe, *xpipeold;
  FieldP          *xp;
  AuxFieldH       *haux;
  AuxFieldP       *paux;
  PetscFunctionBegin;
  
  /* get the residual array */
  ierr = DMDAVecGetArray(user->daPipe,RPipe,&rpipe);CHKERRQ(ierr);
  
  /* get the current guess of solution */
  ierr = DMDAVecGetArrayRead(user->daPipe,XPipe,&xpipe);CHKERRQ(ierr);
  
  ierr = DMDAVecGetArray(user->daP,user->XP,&xp);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daPaux,user->Paux,&paux);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daHaux,user->Haux,&haux);CHKERRQ(ierr);
  
  ierr = DMDAVecGetArray(user->daPipe,user->XPipeold,&xpipeold);CHKERRQ(ierr);
  
  /* get dimensions of LOCAL piece of grid (in parallel, this is not full grid) */
  ierr = DMDAGetGridInfo(user->daP,&is,0,0,&ie,0,0,0,0,0,0);CHKERRQ(ierr);
  
  /* apply boundary condition at r=0 */
  if (is==0) {
    i = 0; is++;
    rpipe[i].qp = xpipe[i].qp;
  }
  
  /* apply boundary condition at r=r_max */
  if (ie==p->ni) {
     i = p->ni-1; ie--;
     rpipe[i].qp = xpipe[i].qp - xpipe[i-1].qp;
  }
  
  /* interior of the domain */
  for (i=is; i<p->ni-1; i++) {
    rpipe[i].qp = qpResidual(user,xpipe,xp,haux,paux,xpipeold,i);
  }
  
  /* clean up */
  ierr = DMDAVecRestoreArrayRead(user->daPipe,XPipe,&xpipe);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daP,user->XP,&xp);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daHaux,user->Haux,&haux);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daPaux,user->Paux,&paux);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daPipe,RPipe,&rpipe);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/* Create the compaction pressure residual */
PetscReal PressureResidual(AppCtx *user, const FieldP *xp, FieldPipe *xpipe, AuxFieldH *haux, AuxFieldP *paux, PetscInt i)
/* ------------------------------------------------------------------- */
{
  Parameter   *p = user->param;
  PetscReal   dr = (1-p->base)/(p->ni-2);
  PetscReal   ft, t, E;
  PetscReal   perm_p, perm_m, div_q, residual;
  
  /* if no porosity, compaction pressure undefined so set to zero */
  if (haux[i].phi == 0) { return xp[i].P; }
  
  /* Use spline interpolation to go from E = 0  to E = linear over region delta above Pc */
  t = (xp[i].P - p->Pc)/p->delta_t;
  if (t<=0) {
    ft = 0;
  } else if (0<t && t<1) {
    ft = 1.0 - pow(1.0-t,2);
  } else {
    ft = 1.0;
  }
  E = p->nu*(xp[i].P - p->Pc) *ft;
  
  /* Permeability at the above and below cell faces, geometric mean */
  perm_p = pow(haux[i].phi*haux[i+1].phi,p->perm/2.0);
  perm_m = pow(haux[i].phi*haux[i-1].phi,p->perm/2.0);
  
  /* div(phi^n *((1-phi) - delta*grad(P) )) */
  div_q = (pow(0.5*(paux[i].r+paux[i+1].r),2)*perm_p*(1.0 - p->phi_0*0.5*(haux[i].phi+haux[i+1].phi) - p->delta*(xp[i+1].P-xp[i].P)/dr)
        -  pow(0.5*(paux[i].r+paux[i-1].r),2)*perm_m*(1.0 - p->phi_0*0.5*(haux[i].phi+haux[i-1].phi) - p->delta*(xp[i].P-xp[i-1].P)/dr))
        / (paux[i].r*paux[i].r*dr);
  
  /* construct residual */
  residual = haux[i].phi*xp[i].P + div_q + E;
  
  return(residual);
}

/* ------------------------------------------------------------------- */
/* Create the enthalpy residual */
PetscReal EnthalpyResidual(AppCtx *user, const FieldH *xh, FieldH *xhold, FieldH *xhtheta, FieldP *xptheta, FieldPipe *xpipetheta, AuxFieldH *hauxtheta, AuxFieldP *pauxtheta, PetscInt i)
/* ------------------------------------------------------------------- */
{
  Parameter   *p = user->param;
  PetscReal   dr = (1.0-p->base)/(p->ni-2);
  PetscReal   v_Fromm_p, v_Fromm_m;
  PetscReal   adv_Ts, adv_Tl, dif_T;
  PetscReal   ft, gphi, t, phi, E, M;
  PetscReal   adv_L;
  PetscReal   residual;
  
  /* Use spline interpolation to go from E = 0  to E = linear over region delta above Pc */
  t = (xptheta[i].P - p->Pc)/p->delta_t;
  if (t<=0) {
    ft = 0;
  } else if (0<t && t<1) {
    ft = 1.0 - pow(1.0-t,2);
  } else {
    ft = 1.0;
  }
  E = p->nu*(xptheta[i].P - p->Pc) *ft;
  
  /* Use spline interpolation to switch off M when qp -> 0, also need cst derivatives in T and Tp */
  phi = xpipetheta[i-1].qp/p->delta_phi;
  if (phi<=0) {
    gphi = 0;
  } else if (0<phi && phi<1) {
    gphi = phi*phi*(3.0-2.0*phi);
  } else {
    gphi = 1.0;
  }
  M = p->hhat*(1.0 - hauxtheta[i].T)*gphi;
  if (hauxtheta[i].T<p->Te) {
    M = 0;
  }
  
  /* advection of sensible heat by the solid: div(uT) */
  adv_Ts = (pow(0.5*(pauxtheta[i].r+pauxtheta[i+1].r),2)*pauxtheta[i].u  *hauxtheta[i+1].T
         -  pow(0.5*(pauxtheta[i].r+pauxtheta[i-1].r),2)*pauxtheta[i-1].u*hauxtheta[i].T)
        /(pauxtheta[i].r*pauxtheta[i].r*dr);
  
  /* advection of sensible heat by the liquid: div((qT) */
  adv_Tl = (pow(0.5*(pauxtheta[i].r+pauxtheta[i+1].r),2)*pauxtheta[i].q  *hauxtheta[i].T
         -  pow(0.5*(pauxtheta[i].r+pauxtheta[i-1].r),2)*pauxtheta[i-1].q*hauxtheta[i-1].T)
        /(pauxtheta[i].r*pauxtheta[i].r*dr);
  
  /* advection of latent heat: div(phi_0*phi*u + q) */
  adv_L = (pow(0.5*(pauxtheta[i].r+pauxtheta[i+1].r),2)*(pauxtheta[i].u  *p->phi_0*0.5*(hauxtheta[i].phi+hauxtheta[i+1].phi) + pauxtheta[i].q)
        -  pow(0.5*(pauxtheta[i].r+pauxtheta[i-1].r),2)*(pauxtheta[i-1].u*p->phi_0*0.5*(hauxtheta[i].phi+hauxtheta[i-1].phi) + pauxtheta[i-1].q))
        / (pauxtheta[i].r*pauxtheta[i].r*dr);
  
  /* heat diffusion: div(grad T) */
  dif_T = (pow(0.5*(pauxtheta[i].r+pauxtheta[i+1].r),2)*(hauxtheta[i+1].T-hauxtheta[i].T  )
        -  pow(0.5*(pauxtheta[i].r+pauxtheta[i-1].r),2)*(hauxtheta[i].T  -hauxtheta[i-1].T))
        / (pauxtheta[i].r*pauxtheta[i].r*dr*dr);
  
  /* Fromm scheme in middle of domain */
  if (i>1 && i<p->ni-2) {
    /* advection of sensible heat by the solid: div(uT) */
    v_Fromm_p = pow(0.5*(pauxtheta[i].r+pauxtheta[i+1].r),2)*pauxtheta[i].u;
    v_Fromm_m = pow(0.5*(pauxtheta[i].r+pauxtheta[i-1].r),2)*pauxtheta[i-1].u;
    adv_Ts = Fromm_advection(user, v_Fromm_p, v_Fromm_m, hauxtheta[i+2].T, hauxtheta[i+1].T, hauxtheta[i].T, hauxtheta[i-1].T, hauxtheta[i-2].T, pauxtheta[i].r);

    /* advection of sensible heat by the liquid: div(qT) */
    v_Fromm_p = pow(0.5*(pauxtheta[i].r+pauxtheta[i+1].r),2)*pauxtheta[i].q;
    v_Fromm_m = pow(0.5*(pauxtheta[i].r+pauxtheta[i-1].r),2)*pauxtheta[i-1].q;
    adv_Tl = Fromm_advection(user, v_Fromm_p, v_Fromm_m, hauxtheta[i+2].T, hauxtheta[i+1].T, hauxtheta[i].T, hauxtheta[i-1].T, hauxtheta[i-2].T, pauxtheta[i].r);
  }
  
  /* Construct residual */
  residual = (xh[i].H - xhold[i].H)/p->dt + adv_Ts + adv_Tl + p->St*adv_L - dif_T/p->Pe - p->St*p->psi + E*(hauxtheta[i].T + p->St) - M*(1.0 + p->St);
  
  return(residual);
}

/* ------------------------------------------------------------------- */
/* Create the plumbing system flux residual */
PetscReal qpResidual(AppCtx *user, const FieldPipe *xpipe, FieldP *xp, AuxFieldH *haux, AuxFieldP *paux, FieldPipe *xpipeold, PetscInt i)
/* ------------------------------------------------------------------- */
{
  Parameter   *p = user->param;
  PetscReal   dr = (1.0-p->base)/(p->ni-2);
  PetscReal   div_qp;
  PetscReal   ft, gphi, t, phi, E, M;
  PetscReal   residual;
  
  /* Use spline interpolation to go from E = 0  to E = linear over region delta above Pc */
  t = (xp[i].P - p->Pc)/p->delta_t;
  if (t<=0) {
    ft = 0;
  } else if (0<t && t<1) {
    ft = 1.0 - pow(1.0-t,2);
  } else {
    ft = 1.0;
  }
  E = p->nu*(xp[i].P - p->Pc) *ft;
  
  /* Use spline interpolation to switch off M when qp -> 0, also need cst derivatives in T and Tp */
  phi = xpipe[i-1].qp/p->delta_phi;
  if (phi<=0) {
    gphi = 0;
  } else if (0<phi && phi<1) {
    gphi = phi*phi*(3.0-2.0*phi);
  } else {
    gphi = 1.0;
  }
  M = p->hhat*(1.0 - haux[i].T) *gphi;
  if (haux[i].T<p->Te) {
    M = 0;
  }
  
  div_qp = (pow(0.5*(paux[i].r+paux[i+1].r),2)*xpipe[i].qp
          - pow(0.5*(paux[i].r+paux[i-1].r),2)*xpipe[i-1].qp)/(paux[i].r*paux[i].r*dr);
  
  residual = div_qp - E + M;
  
  return(residual);
}

/* ------------------------------------------------------------------- */
/* Calculates the auxilliary parameters associated with the pressure solver */
PetscErrorCode PAuxParamsCalc(AppCtx *user, const FieldP *xp, FieldPipe *xpipe, AuxFieldH *haux, AuxFieldP *paux)
/* ------------------------------------------------------------------- */
{
  Parameter   *p = user->param;
  PetscReal   dr = (1.0-p->base)/(p->ni-2);
  PetscInt    i;
  PetscReal   u_max=0;
  PetscFunctionBegin;
  
  for(i=0; i<p->ni; i++) {
    /* set Darcy flux from Darcy's Law, q = (phi_0*phi)^n *(1-phi_0*phi - delta grad(P)) */
    /* avoid overrunning end of arrays by setting surface q = 0 */
    if (i!=p->ni-1) {
      paux[i].q = pow(haux[i].phi*haux[i+1].phi,p->perm/2)*(1.0 - p->phi_0*0.5*(haux[i].phi+haux[i+1].phi) - p->delta*(xp[i+1].P-xp[i].P)/dr);
    } else {
      paux[i].q = 0;
    }
    
    /* solid velocity from continuity, only true in 1D */
    paux[i].u = -(xpipe[i].qp + paux[i].q);
    
    if(fabs(paux[i].u)>u_max) {
      u_max = fabs(paux[i].u);
      p->CFL = u_max*(p->psi_0*p->R/(p->rho_0*p->L))*p->dt*(p->rho_0*p->L/p->psi_0)/(dr*p->R);
    }

  }
  
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/* Calculates the auxilliary parameters associated with the enthalpy solver */
PetscErrorCode HAuxParamsCalc(AppCtx *user, const FieldH *xh, AuxFieldH *haux)
/* ------------------------------------------------------------------- */
{
  Parameter   *p = user->param;
  PetscInt    i;
  PetscFunctionBegin;
  
  for(i=0; i<p->ni; i++) {
    /* set temperature and porosity from enthalpy */
    if (xh[i].H > 1.0) {
      haux[i].T = 1.0;
      haux[i].phi = (xh[i].H-1.0)/(p->St*p->phi_0);
    } else {
      
      haux[i].T = xh[i].H;
      haux[i].phi = 0;
    }
  }
  
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/* Check periodically whether steady state has been achieved */
PetscBool SteadyStateChecker(AppCtx *user)
/* ------------------------------------------------------------------- */
{
  Parameter   *p = user->param;
  PetscBool   STEADY = PETSC_TRUE;
  Vec         XP_sum, XH_sum, XPipe_sum;
  PetscReal   XP_L2, XH_L2, XPipe_L2;
  PetscScalar a = -1, b = 1;
  PetscErrorCode ierr;
  
  ierr = VecDuplicate(user->XP,&XP_sum); CHKERRQ(ierr);
  ierr = VecCopy(user->XP,XP_sum); CHKERRQ(ierr);
  ierr = VecDuplicate(user->XH,&XH_sum); CHKERRQ(ierr);
  ierr = VecCopy(user->XH,XH_sum); CHKERRQ(ierr);
  ierr = VecDuplicate(user->XPipe,&XPipe_sum); CHKERRQ(ierr);
  ierr = VecCopy(user->XPipe,XPipe_sum); CHKERRQ(ierr);
  
  ierr = VecAXPBY(XP_sum,a,b,user->XPold);CHKERRQ(ierr);
  ierr = VecAXPBY(XH_sum,a,b,user->XHold);CHKERRQ(ierr);
  ierr = VecAXPBY(XPipe_sum,a,b,user->XPipeold);CHKERRQ(ierr);
  
  ierr = VecNorm(XP_sum,NORM_2,&XP_L2);CHKERRQ(ierr);
  ierr = VecNorm(XH_sum,NORM_2,&XH_L2);CHKERRQ(ierr);
  ierr = VecNorm(XP_sum,NORM_2,&XPipe_L2);CHKERRQ(ierr);
  
  PetscPrintf(user->comm,"SS checker - XP_L2 = %e, XH_L2 = %e, XPipe_L2 = %e \n",XP_L2,XH_L2,XPipe_L2);
  if (XP_L2 < p->steady_tol && XH_L2 < p->steady_tol && XPipe_L2 < p->steady_tol) {
    STEADY = PETSC_TRUE;
  } else {
    STEADY = PETSC_FALSE;
  }
  
  return(STEADY);
}

/* ------------------------------------------------------------------- */
/* Calculates a flux divergence term using the upwind Fromm scheme (spherical) */
PetscReal Fromm_advection(AppCtx *user, PetscReal v_p, PetscReal v_m,
        PetscReal q_p2, PetscReal q_p, PetscReal q_c,
        PetscReal q_m, PetscReal q_m2, PetscReal r)
/* ------------------------------------------------------------------- */
{
  Parameter   *p = user->param;
  PetscReal   dr = (1.0-p->base)/(p->ni-2);
  PetscReal   adv_term;
  
  adv_term = ( (v_p/8 *(-q_p2 + 5*(q_p + q_c) - q_m ) - fabs(v_p)/8 *(-q_p2 + 3*(q_p - q_c) + q_m) )
     -   (v_m/8 *(-q_p  + 5*(q_c + q_m) - q_m2) - fabs(v_m)/8 *(-q_p  + 3*(q_c - q_m) + q_m2)))/(dr*r*r);
  
  return(adv_term);
}

/* ------------------------------------------------------------------- */
/* Calculate the melting rate, E, and M for outputting and plotting    */
PetscErrorCode Extra_output_params(AppCtx *user)
/* ------------------------------------------------------------------- */
{
  Parameter *p = user->param;
  FieldH  *xh;
  FieldP  *xp;
  FieldPipe *xpipe;
  AuxFieldH *haux, *hauxold;
  AuxFieldP *paux;
  FieldOUT  *xout;
  PetscInt  i;
  PetscReal dr = (1.0-p->base)/(p->ni-2);
  PetscReal ft, ftau, gphi, t, tau, phi;
  PetscErrorCode ierr;
  
  ierr = DMDAVecGetArray(user->daH,user->XH,&xh);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daP,user->XP,&xp);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daPipe,user->XPipe,&xpipe);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daHaux,user->Haux,&haux);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daHaux,user->Hauxold,&hauxold);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daPaux,user->Paux,&paux);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daOUT,user->XOUT,&xout);CHKERRQ(ierr);
  
  for (i=0;i<p->ni;i++) {
    /* Use spline interpolation to go from E = 0  to E = linear over region delta above Pc */
    t = (xp[i].P - p->Pc)/p->delta_t;
    if (t<=0) {
      ft = 0;
    } else if (0<t && t<1) {
      ft = 1.0 - pow(1.0-t,2);
    } else {
      ft = 1.0;
    }
    xout[i].E = p->nu*(xp[i].P - p->Pc) *ft;
    
    /* Use spline interpolation to switch off M when qp -> 0, also need cst derivatives in T and Tp */
    tau = (haux[i].T-p->Te)/p->delta_tau;
    if (tau<=0) {
      ftau = 0;
    } else if (0<tau && tau<1) {
      ftau = tau*tau*(3.0-2.0*tau);
    } else {
      ftau = 1.0;
    }
    phi = xpipe[i-1].qp/p->delta_phi;
    if (phi<=0) {
      gphi = 0;
    } else if (0<phi && phi<1) {
      gphi = phi*phi*(3.0-2.0*phi);
    } else {
      gphi = 1.0;
    }
    xout[i].M = p->hhat*(1.0 - haux[i].T)*gphi;
    if (haux[i].T<p->Te) {
      xout[i].M = 0;
    }

    /* calculate the melting rates */
    if (i>0 && i<p->ni-1 && haux[i].phi > 1e-8 && haux[i+1].phi > 1e-8) {
      xout[i].Gamma = p->phi_0*(haux[i].phi-hauxold[i].phi)/p->dt +
                      (pow(0.5*(paux[i].r+paux[i+1].r),2.0)*(p->phi_0*0.5*(haux[i].phi+haux[i+1].phi)*paux[i].u + paux[i].q) -
                       pow(0.5*(paux[i].r+paux[i-1].r),2.0)*(p->phi_0*0.5*(haux[i].phi+haux[i-1].phi)*paux[i-1].u + paux[i-1].q))/(pow(paux[i].r,2.0)*dr) + xout[i].E;

    } else {
      xout[i].Gamma = 0; // melting rate will be zero at end nodes (as outside the domain)
      xout[i].Gamma_A = 0;
    }
    
  }
  
  ierr = DMDAVecRestoreArray(user->daH,user->XH,&xh);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daP,user->XP,&xp);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daPipe,user->XPipe,&xpipe);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daHaux,user->Haux,&haux);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daHaux,user->Hauxold,&hauxold);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daPaux,user->Paux,&paux);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daOUT,user->XOUT,&xout);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/* Set up parameter bag */
PetscErrorCode ParameterSetup(AppCtx user)
/* ------------------------------------------------------------------- */
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  /* GRID PARAMETERS */
  ierr = PetscBagRegisterInt(user.bag,&(user.param->ni),10000,"ni","Number of grid points"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(user.bag,&(user.param->ns),100000000,"ns","Number of time steps"); CHKERRQ(ierr);
  
  /* IO STRUCTURE PARAMETERS */
  ierr = PetscBagRegisterReal(user.bag,&(user.param->R_cmb),700000,"R_cmb","Core-mantle boundary (base of domain)"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->R),1820000,"R","Radius of Io"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->base),0,"base","Non-dim base of domain, set in initialsation"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->g),1.5,"g","constant gravity"); CHKERRQ(ierr);
  
  /* IO MATERIAL PARAMETERS */
  ierr = PetscBagRegisterReal(user.bag,&(user.param->perm),3,"perm","Porosity exponent"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->L),400000,"L","Latent heat of fusion (J/kg)"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->ce),1200,"ce","specific heat capacity (J/kg/K)"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->K_0),1e-7,"K_0","Mobility"); CHKERRQ(ierr); // from Katz 2010
  ierr = PetscBagRegisterReal(user.bag,&(user.param->eta),1e20,"eta","reference shear viscosity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->eta_l),1,"eta_l","reference liquid viscosity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->rho_0),3000,"rho_0","density of mantle"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->del_rho),500,"del_rho","Boussinesq density difference"); CHKERRQ(ierr); // from Katz 2010
  ierr = PetscBagRegisterReal(user.bag,&(user.param->kappa),1e-6,"kappa","thermal diffusivity"); CHKERRQ(ierr); //value from Katz 2010
  ierr = PetscBagRegisterReal(user.bag,&(user.param->zeta_0),0,"zeta_0","Reference compaction viscosity, set in initialsation"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->D),1e-8,"D","Chemical diffusivity (m2/s)"); CHKERRQ(ierr); //value from Katz 2010
  
  /* IO NON-DIMENSIONAL PARAMETERS */
  ierr = PetscBagRegisterReal(user.bag,&(user.param->Pe),0,"Pe","Peclet number, set in initialsation"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->St),0,"St","Stefan number, set in initialsation"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->delta),0,"delta","Compaction parameter, set in initialsation"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->phi_0),0,"phi_0","Reference porosity, necessarily set by reference velocity and hence by tidal heating, set in initialsation"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->nu),1000,"nu","The constant that sets how rapidly melt is extracted"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->delta_t),0.01,"delta_t","Associated with the pressure switch on extraction"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->delta_tau),0.01,"delta_tau","Associated with the temperatire switch on emplacement"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->delta_phi),0.01,"delta_phi","Associated with the flux switch on emplacement"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->hhat),100,"hhat","Emplacement rate constant"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->Pc),0.1,"Pc","Critical overpressure above which extraction occurs"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->P_0),0,"P_0","Reference compaction pressure, calculated in initialisation"); CHKERRQ(ierr);
  
  /* TEMPERATURE AND HEATING PARAMETERS */
  ierr = PetscBagRegisterReal(user.bag,&(user.param->T_0),1500-150,"T_0","reference temperature"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->T_surf_dim),150,"T_surf_dim","average surface temperature"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->Te),0,"Te","The emplacement cut-off temperature"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->Psi_0),1e14,"Psi_0","Input global tidal heating"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->psi_0),0,"psi_0","Reference constant volumetric heating, set in initialsation"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->psi),1,"psi","non-dimensional tidal heating rate"); CHKERRQ(ierr);
  
  /* BOUNDARY CONDITION PARAMETERS */
  ierr = PetscBagRegisterReal(user.bag,&(user.param->H_end),0,"H_end","BC for enthalpy at top of domain"); CHKERRQ(ierr);
  
  /* TIMESTEPPING PARAMETERS */
  ierr = PetscBagRegisterReal(user.bag,&(user.param->t),0,"t","(NO SET) Current time");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->dt),5e-4,"dt","Time-step size");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->CFL),0,"CFL","reporting CFL value");CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(user.bag,&(user.param->it),1,"it","holds what timestep we're at"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(user.bag,&(user.param->SNESit),15,"SNESit","controls how many iterations are done within a timestep"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->tmax),1e10,"tmax","Maximum model time");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->theta),1,"theta","For theta method, 0.5 = CN, 1 = Implcit, 0 = explicit");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->steady_tol),1e-8,"steady_tol","Tolerance for deciding if steady state has been achieved");CHKERRQ(ierr);
  
  /* OUTPUT FILE PARAMETERS */
  ierr = PetscBagRegisterInt(user.bag,&(user.param->out_freq),100,"out_freq","how often to output solution, default 0 = only at end"); CHKERRQ(ierr);
  ierr = PetscBagRegisterString(user.bag,&(user.param->filename),FNAME_LENGTH,"Io_compaction","filename","Name of output file"); CHKERRQ(ierr);
  ierr = PetscBagRegisterString(user.bag,&(user.param->output_dir),PETSC_MAX_PATH_LEN-1,"outputs/","output_dir","Name of output directory"); CHKERRQ(ierr);
  ierr = PetscBagRegisterString(user.bag,&(user.param->SS_output_dir),PETSC_MAX_PATH_LEN-1,"SS_outputs/","SS_output_dir","Name of output directory"); CHKERRQ(ierr);
  
  /* RESTART PARAMETERS */
  ierr = PetscBagRegisterInt(user.bag,&(user.param->restart_step),0,"restart_step","Step to restart from, 0 = start new, -1 = from steady state file"); CHKERRQ(ierr);
  ierr = PetscBagRegisterString(user.bag,&(user.param->restart_ss),PETSC_MAX_PATH_LEN-1,"SSc2_ha005_hb005_Pc01_Te00","restart_ss","Name of steady state file to restart from"); CHKERRQ(ierr);
  
                                
  /* report contents of parameter structure */
  ierr = PetscPrintf(user.comm,"--------------------------------------\n"); CHKERRQ(ierr);
  ierr = PetscBagView(user.bag,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  ierr = PetscPrintf(user.comm,"--------------------------------------\n"); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/* Create DMDA vectors and sets up SNES */
PetscErrorCode DMDASNESsetup(AppCtx *user)
/* ------------------------------------------------------------------- */
{
  PetscErrorCode ierr;
  PetscInt       dofs;
  PetscFunctionBegin;
  
  /* set up solution and residual vectors for the pressure */
  dofs = (PetscInt)(sizeof(FieldP)/sizeof(PetscReal));
  ierr = DMDACreate1d(user->comm,DM_BOUNDARY_NONE,user->param->ni,dofs,2,PETSC_NULL,&user->daP); CHKERRQ(ierr);
  ierr = DMSetUp(user->daP); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user->daP,0,"P"); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(user->daP,&user->XP); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)user->XP,"grid"); CHKERRQ(ierr);
  ierr = VecDuplicate(user->XP,&user->RP); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)user->RP,"res"); CHKERRQ(ierr);
  ierr = VecDuplicate(user->XP,&user->XPold); CHKERRQ(ierr);
  ierr = VecDuplicate(user->XP,&user->XPtheta); CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)user->XP,"xp_"); CHKERRQ(ierr); // unique prefix to vector that enables searching for its options
  ierr = PetscObjectSetOptionsPrefix((PetscObject)user->RP,"rp_"); CHKERRQ(ierr); // unique prefix to vector that enables searching for its options
  
  /* set up solution and residual vectors for enthalpy */
  dofs = (PetscInt)(sizeof(FieldH)/sizeof(PetscReal));
  ierr = DMDACreate1d(user->comm,DM_BOUNDARY_NONE,user->param->ni,dofs,2,PETSC_NULL,&user->daH); CHKERRQ(ierr);
  ierr = DMSetUp(user->daH); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user->daH,0,"H"); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(user->daH,&user->XH); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)user->XH,"grid"); CHKERRQ(ierr);
  ierr = VecDuplicate(user->XH,&user->RH); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)user->RH,"res"); CHKERRQ(ierr);
  ierr = VecDuplicate(user->XH,&user->XHold); CHKERRQ(ierr);
  ierr = VecDuplicate(user->XH,&user->XHtheta); CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)user->XH,"xh_"); CHKERRQ(ierr); // unique prefix to vector that enables searching for its options
  ierr = PetscObjectSetOptionsPrefix((PetscObject)user->RH,"rh_"); CHKERRQ(ierr); // unique prefix to vector that enables searching for its options
  
  /* set up solution and residual vectors for the pipe */
  dofs = (PetscInt)(sizeof(FieldPipe)/sizeof(PetscReal));
  ierr = DMDACreate1d(user->comm,DM_BOUNDARY_NONE,user->param->ni,dofs,2,PETSC_NULL,&user->daPipe); CHKERRQ(ierr);
  ierr = DMSetUp(user->daPipe); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user->daPipe,0,"qp"); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(user->daPipe,&user->XPipe); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)user->XPipe,"grid"); CHKERRQ(ierr);
  ierr = VecDuplicate(user->XPipe,&user->RPipe); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)user->RPipe,"res"); CHKERRQ(ierr);
  ierr = VecDuplicate(user->XPipe,&user->XPipeold); CHKERRQ(ierr);
  ierr = VecDuplicate(user->XPipe,&user->XPipetheta); CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)user->XPipe,"xpipe_"); CHKERRQ(ierr); // unique prefix to vector that enables searching for its options
  ierr = PetscObjectSetOptionsPrefix((PetscObject)user->RPipe,"rpipe_"); CHKERRQ(ierr); // unique prefix to vector that enables searching for its options
  
  /* set up vector of auxilliary variables */
  dofs = (PetscInt)(sizeof(AuxFieldP)/sizeof(PetscReal));
  ierr = DMDACreate1d(user->comm,DM_BOUNDARY_NONE,user->param->ni,dofs,2,PETSC_NULL,&user->daPaux); CHKERRQ(ierr);
  ierr = DMSetUp(user->daPaux); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user->daPaux,0,"q"); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user->daPaux,1,"u"); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user->daPaux,2,"r"); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(user->daPaux,&user->Paux); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)user->Paux,"grid"); CHKERRQ(ierr);
  ierr = VecDuplicate(user->Paux,&user->Pauxold); CHKERRQ(ierr);
  ierr = VecDuplicate(user->Paux,&user->Pauxtheta); CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)user->Paux,"paux_"); CHKERRQ(ierr); // unique prefix to vector that enables searching for its options
  ierr = PetscObjectSetOptionsPrefix((PetscObject)user->Paux,"paux_"); CHKERRQ(ierr); // unique prefix to vector that enables searching for its options

  /* set up vector of auxilliary variables */
  dofs = (PetscInt)(sizeof(AuxFieldH)/sizeof(PetscReal));
  ierr = DMDACreate1d(user->comm,DM_BOUNDARY_NONE,user->param->ni,dofs,2,PETSC_NULL,&user->daHaux); CHKERRQ(ierr);
  ierr = DMSetUp(user->daHaux); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user->daHaux,0,"phi"); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user->daHaux,1,"T"); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(user->daHaux,&user->Haux); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)user->Haux,"grid"); CHKERRQ(ierr);
  ierr = VecDuplicate(user->Haux,&user->Hauxold); CHKERRQ(ierr);
  ierr = VecDuplicate(user->Haux,&user->Hauxtheta); CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)user->Haux,"haux_"); CHKERRQ(ierr); // unique prefix to vector that enables searching for its options
  ierr = PetscObjectSetOptionsPrefix((PetscObject)user->Haux,"haux_"); CHKERRQ(ierr); // unique prefix to vector that enables searching for its options

  /* set up vector of additional output variables */
  dofs = (PetscInt)(sizeof(FieldOUT)/sizeof(PetscReal));
  ierr = DMDACreate1d(user->comm,DM_BOUNDARY_NONE,user->param->ni,dofs,2,PETSC_NULL,&user->daOUT); CHKERRQ(ierr);
  ierr = DMSetUp(user->daOUT); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user->daOUT,0,"Gamma"); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user->daOUT,1,"Gamma_A"); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user->daOUT,2,"E"); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user->daOUT,3,"M"); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(user->daOUT,&user->XOUT); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)user->XOUT,"grid"); CHKERRQ(ierr);
  
  /* set up nonlinear solver context for pressure equation */
  ierr = SNESCreate(user->comm,&user->snesP);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(user->snesP,"P_");CHKERRQ(ierr);
  ierr = SNESSetDM(user->snesP,user->daP);CHKERRQ(ierr);
  ierr = SNESSetFunction(user->snesP,user->RP,FormResidualP,user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(user->snesP);CHKERRQ(ierr);
  
  /* set up nonlinear solver context for enthalpy */
  ierr = SNESCreate(user->comm,&user->snesH);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(user->snesH,"H_");CHKERRQ(ierr);
  ierr = SNESSetDM(user->snesH,user->daH);CHKERRQ(ierr);
  ierr = SNESSetFunction(user->snesH,user->RH,FormResidualH,user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(user->snesH);CHKERRQ(ierr);
  
  /* set up nonlinear solver context for pipe */
  ierr = SNESCreate(user->comm,&user->snesPipe);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(user->snesPipe,"pipe_");CHKERRQ(ierr);
  ierr = SNESSetDM(user->snesPipe,user->daPipe);CHKERRQ(ierr);
  ierr = SNESSetFunction(user->snesPipe,user->RPipe,FormResidualPipe,user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(user->snesPipe);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/* Clean up by destroying vectors etc */
PetscErrorCode CleanUp(AppCtx *user)
/* ------------------------------------------------------------------- */
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  /* clean up by destroying objects that were created */
  ierr = VecDestroy(&user->XP);CHKERRQ(ierr);
  ierr = VecDestroy(&user->RP);CHKERRQ(ierr);
  ierr = VecDestroy(&user->XPold);CHKERRQ(ierr);
  ierr = VecDestroy(&user->XPtheta);CHKERRQ(ierr);
  ierr = VecDestroy(&user->XH);CHKERRQ(ierr);
  ierr = VecDestroy(&user->RH);CHKERRQ(ierr);
  ierr = VecDestroy(&user->XHold);CHKERRQ(ierr);
  ierr = VecDestroy(&user->XHtheta);CHKERRQ(ierr);
  ierr = VecDestroy(&user->XPipe);CHKERRQ(ierr);
  ierr = VecDestroy(&user->RPipe);CHKERRQ(ierr);
  ierr = VecDestroy(&user->XPipeold);CHKERRQ(ierr);
  ierr = VecDestroy(&user->XPipetheta);CHKERRQ(ierr);
  ierr = VecDestroy(&user->Haux);CHKERRQ(ierr);
  ierr = VecDestroy(&user->Paux);CHKERRQ(ierr);
  ierr = VecDestroy(&user->Hauxold);CHKERRQ(ierr);
  ierr = VecDestroy(&user->Pauxold);CHKERRQ(ierr);
  ierr = VecDestroy(&user->Hauxtheta);CHKERRQ(ierr);
  ierr = VecDestroy(&user->Pauxtheta);CHKERRQ(ierr);
  ierr = SNESDestroy(&user->snesP);CHKERRQ(ierr);
  ierr = SNESDestroy(&user->snesH);CHKERRQ(ierr);
  ierr = DMDestroy(&user->daP);CHKERRQ(ierr);
  ierr = DMDestroy(&user->daH);CHKERRQ(ierr);
  ierr = DMDestroy(&user->daHaux);CHKERRQ(ierr);
  ierr = DMDestroy(&user->daPaux);CHKERRQ(ierr);
  ierr = PetscBagDestroy(&user->bag);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return(ierr);
}

/* ------------------------------------------------------------------- */
PetscErrorCode projCreateDirectory(const char dirname[])
/* This generates a new directory called dirname. If dirname already exists,
   nothing happens. Importantly,
   a) only rank 0 tries to create the directory.
   b) nested directorys cannot be created, e.g. dirname[] = "output/step0/allmydata"
   is not valid. Instead you would have to call the function 3 times
   projCreateDirectory("output/");
   projCreateDirectory("output/step0");
   projCreateDirectory("output/step0/allmydata")
   ** Writen by Dave May, ask him if help needed **; */
/* ------------------------------------------------------------------- */
{
  PetscMPIInt rank;
  int num,error_number;
  PetscBool proj_log = PETSC_FALSE;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  
  /* Let rank 0 create a new directory on proc 0 */
  if (rank == 0) {
    num = mkdir(dirname,S_IRWXU);
    error_number = errno;
  }
  ierr = MPI_Bcast(&num,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(&error_number,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  
  if (error_number == EEXIST) {
    if (proj_log) PetscPrintf(PETSC_COMM_WORLD,"[proj] Writing output to existing directory: %s\n",dirname);
  } else if (error_number == EACCES) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"[proj] Write permission is denied for the parent directory in which the new directory is to be added");
  } else if (error_number == EMLINK) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"[proj] The parent directory has too many links (entries)");
  } else if (error_number == ENOSPC) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"[proj] The file system doesn't have enough room to create the new directory");
  } else if (error_number == ENOSPC) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"[proj] The parent directory of the directory being created is on a read-only file system and cannot be modified");
  } else {
    if (proj_log) PetscPrintf(PETSC_COMM_WORLD,"[proj] Created output directory: %s\n",dirname);
  }
  
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/* To make an output file */
PetscErrorCode DoOutput(AppCtx *user, int it, PetscBool STEADY)
/* ------------------------------------------------------------------- */
{
  Parameter      *p = user->param;
  char*          filename = NULL;
  PetscInt       hhat = p->hhat, Pc = p->Pc*100, Te = p->Te*10;
  PetscViewer    viewer;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  if (!STEADY) {
    asprintf(&filename,"%s%s_%04d",p->output_dir,p->filename,it);
  } else {
    asprintf(&filename,"%sSS_h%03d_Pc%03d_Te%02d",p->SS_output_dir,hhat,Pc,Te);
    
  }
  
  ierr = Extra_output_params(user);CHKERRQ(ierr); // calculate extra outputting parameters
  
  ierr = PetscPrintf(user->comm," Generating output file: \"%s\"\n",filename);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(user->comm,filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = PetscBagView(user->bag,viewer);CHKERRQ(ierr); // output bag
  ierr = VecView(user->XP,viewer);CHKERRQ(ierr);        // output solution
  ierr = VecView(user->XH,viewer);CHKERRQ(ierr);        // output solution
  ierr = VecView(user->XPipe,viewer);CHKERRQ(ierr);        // output solution
  ierr = VecView(user->XOUT,viewer);CHKERRQ(ierr);        // output solution
  ierr = VecView(user->Haux,viewer);CHKERRQ(ierr);     // output auxilliary/diagnostic variables
  ierr = VecView(user->Paux,viewer);CHKERRQ(ierr);     // output auxilliary/diagnostic variables
  ierr = VecView(user->RP,viewer);CHKERRQ(ierr);        // output residual
  ierr = VecView(user->RH,viewer);CHKERRQ(ierr);        // output residual
  ierr = VecView(user->RPipe,viewer);CHKERRQ(ierr);        // output residual
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  free(filename);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/* Import state from an output file */
PetscErrorCode RestartFromFile(AppCtx *user)
/* ------------------------------------------------------------------- */
{
  Parameter      *p = user->param;
  char*          filename = NULL;
  PetscViewer    viewer;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  asprintf(&filename,"%s%s_%04d",p->output_dir,p->filename,p->restart_step);
  if (p->restart_step == -1) {
    asprintf(&filename,"%s%s",p->SS_output_dir,p->restart_ss);
  }
  
  ierr = PetscPrintf(user->comm," Restarting from file: \"%s\"\n",filename);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(user->comm,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = PetscBagLoad(viewer,user->bag);CHKERRQ(ierr); // read in bag
  ierr = VecLoad(user->XP,viewer);CHKERRQ(ierr);  // read in solution
  ierr = VecLoad(user->XH,viewer);CHKERRQ(ierr);  // read in solution
  ierr = VecLoad(user->XPipe,viewer);CHKERRQ(ierr); // read in solution
  ierr = VecLoad(user->XOUT,viewer);CHKERRQ(ierr); // read in solution
  ierr = VecLoad(user->Haux,viewer);CHKERRQ(ierr);  // read in auxilliary/diagnostic variables
  ierr = VecLoad(user->Paux,viewer);CHKERRQ(ierr);  // read in auxilliary/diagnostic variables
  ierr = VecLoad(user->RP,viewer);CHKERRQ(ierr);  // read in residual
  ierr = VecLoad(user->RH,viewer);CHKERRQ(ierr);  // read in residual
  ierr = VecLoad(user->RPipe,viewer);CHKERRQ(ierr);  // read in residual
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  free(filename);
  
  ierr = PetscBagSetFromOptions(user->bag);CHKERRQ(ierr); // take any new options from command line
  
  /* Re-calculate reference values and non-dim values */
  p->psi_0 = p->Psi_0/(4.0/3.0 * PETSC_PI * (pow(p->R,3)-pow(p->R_cmb,3))); // set reference constant heating rate in W/m3
  p->Pe = p->psi_0*p->R*p->R/(p->L*p->kappa*p->rho_0); // Peclet number
  p->St = p->L/(p->ce*p->T_0); // Stefan number
  p->phi_0 = pow(p->psi_0*p->R*p->eta_l/(p->L*p->rho_0*p->K_0*p->del_rho*p->g),1/p->perm); // reference porosity
  p->zeta_0 = p->eta/p->phi_0;
  p->delta = p->zeta_0*p->K_0*pow(p->phi_0,p->perm)/(p->eta_l*p->R*p->R);
  p->P_0 = p->zeta_0*p->psi_0/(p->rho_0*p->L); // reference pressure
  
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/* Get useful information from the DMDA */
PetscErrorCode DMDAGetGridInfo(DM da, int *is, int *js, int *ks, int *ie,
             int *je, int *ke, int *ni, int *nj, int *nk,
             int *dim)
/* ------------------------------------------------------------------- */
{
  PetscErrorCode ierr;
  PetscInt       im, jm, km;
  PetscFunctionBegin;
  ierr = DMDAGetCorners(da,is,js,ks,&im,&jm,&km); CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,dim,ni,nj,nk,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  if (ie) *ie = *is + im;
  if (je) *je = *js + jm;
  if (ke) *ke = *ks + km;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/* Gets array from a vector associated with a DMDA, with ghost points */
PetscErrorCode DAGetGhostedArray(DM da, Vec globvec, Vec *locvec, void *arr)
/* ------------------------------------------------------------------- */
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DMGetLocalVector(da,locvec);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,globvec,INSERT_VALUES,*locvec);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (da,globvec,INSERT_VALUES,*locvec);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,*locvec,arr); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/* Restores array from a vector associated with a DMDA, with ghost points */
PetscErrorCode DARestoreGhostedArray(DM da, Vec globvec, Vec *locvec, void *arr)
/* ------------------------------------------------------------------- */
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DMDAVecRestoreArray(da,*locvec,arr); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(da,*locvec,INSERT_VALUES,globvec);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (da,*locvec,INSERT_VALUES,globvec);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,locvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

