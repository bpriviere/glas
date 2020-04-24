/**
 * controller_sjc.h - Controller Interface for nonlinear controller
 */
#ifndef __CONTROLLER_SJC_H__
#define __CONTROLLER_SJC_H__

#include "stabilizer_types.h"

void controllerSJCInit(void);
bool controllerSJCTest(void);
void controllerSJC(control_t *control, setpoint_t *setpoint,
                    const sensorData_t *sensors,
                    const state_t *state,
                    const uint32_t tick);

struct vec* controllerSJCGetqr(void);
struct vec* controllerSJCGetq(void);
struct vec* controllerSJCGetomega(void);
struct vec* controllerSJCGetomegar(void);

#endif //__CONTROLLER_SJC_H__
