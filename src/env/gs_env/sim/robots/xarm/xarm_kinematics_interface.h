#ifndef __XARM_KINEMATICS_USER_INTERFACE__
#define __XARM_KINEMATICS_USER_INTERFACE__
#include <stddef.h>
#include <vector>
/* ERROR code at return : */
#define KINEMATICS_NOT_CONFIGURED 999

/* Default joint ranges: */
#define XARM7_ANGLE_MAX \
  { 2.0 * M_PI, 2.0944, 2.0 * M_PI, 3.927, 2.0 * M_PI, M_PI, 2.0 * M_PI }
#define XARM7_ANGLE_MIN \
  { -2.0 * M_PI, -2.059, -2.0 * M_PI, -0.19198, -2.0 * M_PI, -1.69297, -2.0 * M_PI }

#define XARM7_ANGLE_MAX_LIMITED \
  { M_PI, 2.0944, M_PI, 3.927, M_PI, M_PI, M_PI }
#define XARM7_ANGLE_MIN_LIMITED \
  { -M_PI, -2.059, -M_PI, -0.19198, -M_PI, -1.69297, -M_PI }

#define XARM6_ANGLE_MAX \
   { 2.0 * M_PI, 2.0944, 0.19198, 2.0 * M_PI, M_PI, 2.0 * M_PI }
#define XARM6_ANGLE_MIN \
   { -2.0 * M_PI, -2.059, -3.927, -2.0 * M_PI, -1.69297, -2.0 * M_PI }

#define XARM6_ANGLE_MAX_LIMITED \
  { M_PI, 2.0944, 0.19198,  M_PI, M_PI, M_PI }
#define XARM6_ANGLE_MIN_LIMITED \
  { -M_PI, -2.059, -3.927, - M_PI, -1.69297, -M_PI }

#define XARM5_ANGLE_MAX \
  { 2.0 * M_PI, 2.0944, 0.19198, M_PI, 2.0 * M_PI }
#define XARM5_ANGLE_MIN \
  { -2.0 * M_PI, -2.059, -3.927, -1.69297, -2.0 * M_PI }

#define UFLITE6_ANGLE_MAX \
  { 2.0 * M_PI, 2.61799, 5.235988, 2.0 * M_PI, 2.1642, 2.0 * M_PI }
#define UFLITE6_ANGLE_MIN \
  { -2.0 * M_PI, -2.61799, -0.061087, -2.0 * M_PI, -2.1642, -2.0 * M_PI }

/* 850mm 6Axis, To Be confirmed */
#define UF850_ANGLE_MAX \
  { 2.0 * M_PI, 2.3038346, 0.061087, 2.0 * M_PI, 2.1642, 2.0 * M_PI }
#define UF850_ANGLE_MIN \
  { -2.0 * M_PI, -2.3038346, -4.2236968, -2.0 * M_PI, -2.1642, -2.0 * M_PI }


/* User APIs: (units: mm for length, radian for angles)
* tcp_mat: TCP in 4x4 homogeneous transformation matrix;
* pose_rpy TCP pose as [x,y,z,roll,pitch,yaw]; 
* theta: joint angles;
* q_pre: reference joint angles for IK.
*/
int xarm5_config(double *q_max, double *q_min, double *tcp_offset=NULL, double *world_offset=NULL);
int xarm5_forward_kinematics(double *theta, double tcp_mat[4][4]);
int xarm5_forward_kinematics(double *theta, double pose_rpy[6]);
int xarm5_inverse_kinematics(double tcp_mat[4][4], double *q_pre, double *theta); 
int xarm5_inverse_kinematics(double pose_rpy[6], double *q_pre, double *theta);

int xarm6_config(double *q_max, double *q_min, double *tcp_offset=NULL, double *world_offset=NULL);
int xarm6_forward_kinematics(double *theta, double tcp_mat[4][4]);
int xarm6_forward_kinematics(double *theta, double pose_rpy[6]);
int xarm6_inverse_kinematics(double tcp_mat[4][4], double *q_pre, double *theta); 
int xarm6_inverse_kinematics(double pose_rpy[6], double *q_pre, double *theta);

int xarm7_config(double *q_max, double *q_min, double *tcp_offset=NULL, double *world_offset=NULL);
int xarm7_forward_kinematics(double *theta, double tcp_mat[4][4]);
int xarm7_forward_kinematics(double *theta, double pose_rpy[6]);
int xarm7_inverse_kinematics(double tcp_mat[4][4], double *q_pre, double *theta); 
int xarm7_inverse_kinematics(double pose_rpy[6], double *q_pre, double *theta);

int uf_lite6_config(double *q_max, double *q_min, double *tcp_offset=NULL, double *world_offset=NULL);
int uf_lite6_forward_kinematics(double *theta, double tcp_mat[4][4]);
int uf_lite6_forward_kinematics(double *theta, double pose_rpy[6]);
int uf_lite6_inverse_kinematics(double tcp_mat[4][4], double *q_pre, double *theta, double* ext_q_max=NULL, double* ext_q_min=NULL, std::vector<std::vector<double> > *all_sol=NULL); 
int uf_lite6_inverse_kinematics(double pose_rpy[6], double *q_pre, double *theta, double* ext_q_max=NULL, double* ext_q_min=NULL, std::vector<std::vector<double> > *all_sol=NULL); 


int uf_850_config(double *q_max, double *q_min, double *tcp_offset=NULL, double *world_offset=NULL);
int uf_850_forward_kinematics(double *theta, double tcp_mat[4][4]);
int uf_850_forward_kinematics(double *theta, double pose_rpy[6]);
int uf_850_inverse_kinematics(double tcp_mat[4][4], double *q_pre, double *theta, double* ext_q_max=NULL, double* ext_q_min=NULL, std::vector<std::vector<double> > *all_sol=NULL); 
int uf_850_inverse_kinematics(double pose_rpy[6], double *q_pre, double *theta, double* ext_q_max=NULL, double* ext_q_min=NULL, std::vector<std::vector<double> > *all_sol=NULL);

/* Axis-Angle pose and Transformation Matrix conversions: */
void axisAngle_to_transMat(double pose_aa[6], double transMat[4][4]);
void transMat_to_axisAngle(double transMat[4][4], double pose_aa[6]);

#endif
