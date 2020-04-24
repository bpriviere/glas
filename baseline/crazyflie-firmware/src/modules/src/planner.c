/*
 *    ______
 *   / ____/________ _____  __  ________      ______ __________ ___
 *  / /   / ___/ __ `/_  / / / / / ___/ | /| / / __ `/ ___/ __ `__ \
 * / /___/ /  / /_/ / / /_/ /_/ (__  )| |/ |/ / /_/ / /  / / / / / /
 * \____/_/   \__,_/ /___/\__, /____/ |__/|__/\__,_/_/  /_/ /_/ /_/
 *                       /____/
 *
 * Crazyswarm advanced control firmware for Crazyflie
 *

The MIT License (MIT)

Copyright (c) 2018 Wolfgang Hoenig and James Alan Preiss

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/*
implementation of planning state machine
*/
#include <stddef.h>
#include "planner.h"

#include "crtp_localization_service.h"
#ifndef SWIG
#include "param.h"
#endif
static uint8_t enableAP = 1;
static float Ko = 0.1; // gain for obstacles (repulsive)
static float Kp = 8; // gain for goal (attractive)
static float Kd = 4;
static float Rsafe = 0.4;
static float max_a = 1.0;//1.0;
static float max_v = 0.25;//25;
static void plan_takeoff_or_landing(struct planner *p, struct vec pos, float yaw, float height, float duration)
{
	struct vec takeoff_pos = pos;
	takeoff_pos.z = height;

	piecewise_plan_7th_order_no_jerk(&p->planned_trajectory, duration,
		pos,         yaw, vzero(), 0, vzero(),
		takeoff_pos,   0, vzero(), 0, vzero());
}

static struct traj_eval artificial_potential(struct planner *p, struct traj_eval input, float t, uint64_t ticks)
{
  if (!enableAP) {
    return input;
  }
#if 0
  // attractive accelleration:
  struct vec a = vadd(vscl(Kp, vclampabs(vsub(input.pos, p->apPos), vrepeat(0.5))),
                      vscl(Kd, vclampabs(vsub(input.vel, p->apVel), vrepeat(0.5))));

  struct allCfState* myState = locSrvGetState(p->my_id);
  for (int id = MIN_CF_ID; id < MAX_CF_ID; ++id) {
    // dx: 0.2, dy: 0.6, dz: 0---0.7

    struct allCfState* otherState = locSrvGetState(id);

    // ignore myself and agents that have not received an update for 500ms
    if (id != p->my_id
        && ticks - otherState->timestamp < 500) {

      struct vec dpos = vsub(otherState->pos, myState->pos);
      float dist = vmag(dpos);
      if (dist < Rsafe) {
        a = vsub(a, vscl(Ko / powf(dist, 3), dpos));
      }
    }
  }
  // if (p->my_id == 1) {
  //   printf("%f,%f,%f\n", a.x, a.y, a.z);
  // }

  // bound force
  a = vclampabs(a, vrepeat(max_a));

  const float dt = t - p->last_t;//1.0 / 500.0f;
  // printf("%f\n", dt);

  // propagate double integrator
  p->apPos = vadd(p->apPos, vscl(dt, p->apVel));
  p->apVel = vclampabs(vadd(p->apVel, vscl(dt, a)), vrepeat(max_v));
  p->last_t = t;

  struct traj_eval ev;
  // ev.pos = vadd(input.pos, vscl(dt * dt, force));
  // ev.vel = vzero();
  ev.pos = p->apPos;
  ev.vel = p->apVel;
  ev.acc = a;
  ev.omega = vzero();
  return ev;
#endif
}

// ----------------- //
// public functions. //
// ----------------- //

void plan_init(struct planner *p, int my_id)
{
	p->state = TRAJECTORY_STATE_IDLE;
	p->reversed = false;
	p->trajectory = NULL;
	p->planned_trajectory.pieces = p->pieces;
  p->my_id = my_id;
  p->apPos = vzero();
  p->apVel = vzero();
}

void plan_stop(struct planner *p)
{
	p->state = TRAJECTORY_STATE_IDLE;
}

bool plan_is_stopped(struct planner *p)
{
	return p->state == TRAJECTORY_STATE_IDLE;
}

struct traj_eval plan_current_goal(struct planner *p, float t, uint64_t ticks)
{
	switch (p->state) {
		case TRAJECTORY_STATE_LANDING:
			if (piecewise_is_finished(p->trajectory, t)) {
				p->state = TRAJECTORY_STATE_IDLE;
			}
			// intentional fall-thru
		case TRAJECTORY_STATE_FLYING:
			if (p->reversed) {
				return artificial_potential(p, piecewise_eval_reversed(p->trajectory, t), t, ticks);
			}
			else {
				return artificial_potential(p, piecewise_eval(p->trajectory, t), t, ticks);
			}

		default:
			return traj_eval_invalid();
	}
}

static struct traj_eval plan_current_goal_noAP(struct planner *p, float t)
{
	switch (p->state) {
		case TRAJECTORY_STATE_LANDING:
			if (piecewise_is_finished(p->trajectory, t)) {
				p->state = TRAJECTORY_STATE_IDLE;
			}
			// intentional fall-thru
		case TRAJECTORY_STATE_FLYING:
			if (p->reversed) {
				return piecewise_eval_reversed(p->trajectory, t);
			}
			else {
				return piecewise_eval(p->trajectory, t);
			}

		default:
			return traj_eval_invalid();
	}
}


int plan_takeoff(struct planner *p, struct vec pos, float yaw, float height, float duration, float t)
{
	if (p->state != TRAJECTORY_STATE_IDLE) {
		return 1;
	}

	plan_takeoff_or_landing(p, pos, yaw, height, duration);
	p->reversed = false;
	p->state = TRAJECTORY_STATE_FLYING;
	p->planned_trajectory.t_begin = t;
	p->trajectory = &p->planned_trajectory;
  p->apPos = pos;
  p->apVel = vzero();
	return 0;
}

int plan_land(struct planner *p, struct vec pos, float yaw, float height, float duration, float t)
{
	if (   p->state == TRAJECTORY_STATE_IDLE
		|| p->state == TRAJECTORY_STATE_LANDING) {
		return 1;
	}

	plan_takeoff_or_landing(p, pos, yaw, height, duration);
	p->reversed = false;
	p->state = TRAJECTORY_STATE_LANDING;
	p->planned_trajectory.t_begin = t;
	p->trajectory = &p->planned_trajectory;
	return 0;
}

int plan_go_to(struct planner *p, bool relative, struct vec hover_pos, float hover_yaw, float duration, float t)
{
	// allow in any state, i.e., can also be used to take-off or land

	struct traj_eval setpoint = plan_current_goal_noAP(p, t);

	if (relative) {
		hover_pos = vadd(hover_pos, setpoint.pos);
		hover_yaw += setpoint.yaw;
	}

	piecewise_plan_7th_order_no_jerk(&p->planned_trajectory, duration,
		setpoint.pos, setpoint.yaw, setpoint.vel, setpoint.omega.z, setpoint.acc,
		hover_pos,    hover_yaw,    vzero(),      0,                vzero());

	p->reversed = false;
	p->state = TRAJECTORY_STATE_FLYING;
	p->planned_trajectory.t_begin = t;
	p->trajectory = &p->planned_trajectory;
	return 0;
}

int plan_start_trajectory( struct planner *p, const struct piecewise_traj* trajectory, bool reversed)
{
	p->reversed = reversed;
	p->trajectory = trajectory;
	p->state = TRAJECTORY_STATE_FLYING;

	return 0;
}

#ifndef SWIG

PARAM_GROUP_START(planner)

PARAM_ADD(PARAM_UINT8, enAP, &enableAP)

PARAM_ADD(PARAM_FLOAT, Ko, &Ko)
PARAM_ADD(PARAM_FLOAT, Kp, &Kp)
PARAM_ADD(PARAM_FLOAT, Kd, &Kd)
PARAM_ADD(PARAM_FLOAT, Rsafe, &Rsafe)
PARAM_ADD(PARAM_FLOAT, max_a, &max_a)
PARAM_ADD(PARAM_FLOAT, max_v, &max_v)
PARAM_GROUP_STOP(planner)

#endif