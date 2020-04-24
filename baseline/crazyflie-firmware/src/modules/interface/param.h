/**
 *    ||          ____  _ __                           
 * +------+      / __ )(_) /_______________ _____  ___ 
 * | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
 * +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
 *  ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
 *
 * Crazyflie control firmware
 *
 * Copyright (C) 2011-2012 Bitcraze AB
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, in version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * param.h - Crazy parameter system header file.
 */

#ifndef __PARAM_H__
#define __PARAM_H__

#include <stdbool.h>
#include <stdint.h>

/* Public functions */
void paramInit(void);
bool paramTest(void);

/* Basic parameter structure */
struct param_s {
  uint8_t type;
  char * name;
  void * address;
};

#define PARAM_BYTES_MASK 0x03
#define PARAM_1BYTE  0x00
#define PARAM_2BYTES 0x01
#define PARAM_4BYTES 0x02
#define PARAM_8BYTES 0x03

#define PARAM_TYPE_INT   (0x00<<2)
#define PARAM_TYPE_FLOAT (0x01<<2)

#define PARAM_SIGNED (0x00<<3)
#define PARAM_UNSIGNED (0x01<<3)

#define PARAM_VARIABLE (0x00<<7)
#define PARAM_GROUP    (0x01<<7)

#define PARAM_RONLY (1<<6)

#define PARAM_START 1
#define PARAM_STOP  0

#define PARAM_SYNC 0x02

// User-friendly macros
#define PARAM_UINT8 (PARAM_1BYTE | PARAM_TYPE_INT | PARAM_UNSIGNED)
#define PARAM_INT8  (PARAM_1BYTE | PARAM_TYPE_INT | PARAM_SIGNED)
#define PARAM_UINT16 (PARAM_2BYTES | PARAM_TYPE_INT | PARAM_UNSIGNED)
#define PARAM_INT16  (PARAM_2BYTES | PARAM_TYPE_INT | PARAM_SIGNED)
#define PARAM_UINT32 (PARAM_4BYTES | PARAM_TYPE_INT | PARAM_UNSIGNED)
#define PARAM_INT32  (PARAM_4BYTES | PARAM_TYPE_INT | PARAM_SIGNED)

#define PARAM_FLOAT (PARAM_4BYTES | PARAM_TYPE_FLOAT | PARAM_SIGNED)

/* Macros */
#define PARAM_ADD(TYPE, NAME, ADDRESS) \
   { .type = TYPE, .name = #NAME, .address = (void*)(ADDRESS), },

#define PARAM_ADD_GROUP(TYPE, NAME, ADDRESS) \
   { \
  .type = TYPE, .name = #NAME, .address = (void*)(ADDRESS), },

// Fix to make unit tests run on MacOS
#ifdef __APPLE__
#define PARAM_GROUP_START(NAME)  \
  static const struct param_s __params_##NAME[] __attribute__((section("__DATA,__.param." #NAME), used)) = { \
  PARAM_ADD_GROUP(PARAM_GROUP | PARAM_START, NAME, 0x0)
#else
#define PARAM_GROUP_START(NAME)  \
  static const struct param_s __params_##NAME[] __attribute__((section(".param." #NAME), used)) = { \
  PARAM_ADD_GROUP(PARAM_GROUP | PARAM_START, NAME, 0x0)
#endif

//#define PARAM_GROUP_START_SYNC(NAME, LOCK) PARAM_ADD_GROUP(PARAM_GROUP | PARAM_START, NAME, LOCK);

#define PARAM_GROUP_STOP(NAME) \
  PARAM_ADD_GROUP(PARAM_GROUP | PARAM_STOP, stop_##NAME, 0x0) \
  };

#endif /* __PARAM_H__ */

