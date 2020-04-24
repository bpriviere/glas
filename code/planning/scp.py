import cvxpy as cp
# import numpy as np
import autograd.numpy as np  # Thinly-wrapped numpy
import scipy
import traceback
from autograd import grad, elementwise_grad, jacobian

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.backends.backend_pdf

# obj is one of "minimizeError", "minimizeX", "minimizeU"
def scp(param, env, x, u, obj, xf = None):

  partialFx = jacobian(env.f_scp, 0)
  partialFu = jacobian(env.f_scp, 1)

  def constructA(xbar, ubar):
    return partialFx(xbar, ubar)

  def constructB(xbar, ubar):
    return partialFu(xbar, ubar)

  # data = np.loadtxt(param.rrt_fn, delimiter=',', ndmin=2)
  # xprev = data[:,0:env.n]
  # uprev = data[:,env.n:env.n + env.m]

  xprev = x
  uprev = u

  T = x.shape[0]
  dt = param.sim_dt

  if xf is None:
    goalState = param.ref_trajectory[:,-1]
  else:
    goalState = xf

  x0 = xprev[0]

  print(np.tile(goalState, (T,1)).shape)

  if param.scp_pdf_fn is not None:
    pdf = matplotlib.backends.backend_pdf.PdfPages(param.scp_pdf_fn)

  objectiveValues = []
  xChanges = []
  uChanges = []
  try:
    # obj = 'minimizeError' # 'minimizeError', 'minimizeX'

    for iteration in range(10):
      print("SCP iteration ", iteration)

      x = cp.Variable((T, env.n))
      u = cp.Variable((T, env.m))

      if obj == 'minimizeError':
        delta = cp.Variable()
        objective = cp.Minimize(delta)
      elif obj == 'minimizeX':
        objective = cp.Minimize(cp.sum_squares(x - np.tile(goalState, (T,1))))
      else:
        # objective = cp.Minimize(cp.sum_squares(u) + 10 * cp.sum_squares(x[:,3:5]))
        objective = cp.Minimize(cp.sum_squares(u))
        # objective = cp.Minimize(cp.sum_squares(u) + 10 * cp.sum(x[:,4]))
      constraints = [
        x[0] == x0, # initial state constraint
      ]

      if obj == 'minimizeError':
        if goalState is not None:
          constraints.append( cp.abs(x[-1] - goalState) <= delta )
        else:
          constraints.append(cp.abs(x[-1,0:2] - goalPos) <= delta )
      elif obj == 'minimizeX':
        pass
      else:
        if goalState is not None:
          constraints.append( x[-1] == goalState )
        else:
          constraints.append( x[-1,0:2] == goalPos )

      # trust region
      for t in range(0, T):
        constraints.append(
          cp.abs(x[t] - xprev[t]) <= 2 #0.1
        )
        constraints.append(
          cp.abs(u[t] - uprev[t]) <= 2 #0.1
        )

      # dynamics constraints
      for t in range(0, T-1):
        xbar = xprev[t]
        ubar = uprev[t]
        A = constructA(xbar, ubar)
        B = constructB(xbar, ubar)
        # print(xbar, ubar, A, B)
        # print(robot.f(xbar, ubar))
        # # simple version:
        constraints.append(
          x[t+1] == x[t] + dt * (env.f_scp(xbar, ubar) + A @ (x[t] - xbar) + B @ (u[t] - ubar))
          )
        # # discretized zero-order hold
        # F = scipy.linalg.expm(A * dt)
        # G = np.zeros(B.shape)
        # H = np.zeros(xbar.shape)
        # for tau in np.linspace(0, dt, 10):
        #   G += (scipy.linalg.expm(A * tau) @ B) * dt / 10
        #   H += (scipy.linalg.expm(A * tau) @ (robot.f(xbar, ubar) - A @ xbar - B @ ubar)) * dt / 10
        # constraints.append(
        #   x[t+1] == F @ x[t] + G @ u[t] + H
        #   )

      # bounds (x and u)
      for t in range(0, T):
        constraints.extend([
          env.s_min <= x[t],
          x[t] <= env.s_max,
          env.a_min <= u[t],
          u[t] <= env.a_max
          ])

      prob = cp.Problem(objective, constraints)

      # The optimal objective value is returned by `prob.solve()`.
      # result = prob.solve(verbose=True,solver=cp.GUROBI, BarQCPConvTol=1e-8)
      try:
        result = prob.solve(verbose=True,solver=cp.GUROBI, BarQCPConvTol=1e-8)
      except cp.error.SolverError:
        return

      if result is None:
        return

      print("Success. Objective: ", result)

      objectiveValues.append(result)
      xChanges.append(np.linalg.norm(x.value - xprev))
      uChanges.append(np.linalg.norm(u.value - uprev))

      # The optimal value for x is stored in `x.value`.
      # print(x.value)
      # print(u.value)

      # compute forward propagated value
      xprop = np.empty(x.value.shape)
      # xprevprop = np.empty(x.value.shape)
      xprop[0] = x0
      # xprevprop[0] = x0
      for t in range(0, T-1):
        xprop[t+1] = xprop[t] + dt * env.f_scp(xprop[t], u.value[t])
        # xprop[t+1] = x.value[t] + dt * env.f_scp(x.value[t], u.value[t])
        # xprevprop[t+1] = xprevprop[t] + dt * env.f_scp(xprevprop[t], uprev[t])

      # print(xprop)
      if param.scp_pdf_fn is not None:
        for i in range(env.n):
          fig, ax = plt.subplots()
          ax.set_title(env.states_name[i])
          ax.plot(xprev[:,i],label="input")
          # ax.plot(xprevprop[:,i],label="input forward prop")
          ax.plot(x.value[:,i],label="opt")
          ax.plot(xprop[:,i],label="opt forward prop")
          ax.legend()
          pdf.savefig(fig)
          plt.close(fig)

        for i in range(env.m):
          fig, ax = plt.subplots()
          ax.set_title(env.actions_name[i])
          ax.plot(uprev[:,i],label="input")
          ax.plot(u.value[:,i],label="opt")
          ax.legend()
          pdf.savefig(fig)
          plt.close(fig)

      xprev = np.array(x.value)
      uprev = np.array(u.value)

      if obj == 'minimizeError' and result < 1e-6:
        break

    if True: #obj == 'minimizeU' or obj == 'minimizeX':
      # result = np.hstack([xprev, uprev])
      # np.savetxt(param.scp_fn, result, delimiter=',')
      return xprev, uprev, objectiveValues[-1]

  except Exception as e:
    print(e)
    traceback.print_tb(e.__traceback__)
  finally:
    # print(xprev)
    # print(uprev)
    if param.scp_pdf_fn is not None:
      fig, ax = plt.subplots()
      try:
        ax.plot(np.arange(1,len(objectiveValues)+1), objectiveValues, '*-', label='cost')
        ax.plot(np.arange(1,len(objectiveValues)+1), xChanges, '*-', label='|x-xp|')
        ax.plot(np.arange(1,len(objectiveValues)+1), uChanges, '*-', label='|u-up|')
      except:
        print("Error during plotting!")
      plt.legend()
      pdf.savefig(fig)
      pdf.close()