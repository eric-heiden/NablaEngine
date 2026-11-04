"""Reach-only scalar symmetric gripper, outside the seven-DOF dense solve."""
def transform(src,tag):
 if tag!='reach':return src
 old='pos[k]+=locked_finger_offset(3*(b-10)+k);'
 assert old in src
 src=src.replace(old,'pos[k]+=locked_finger_offset(3*(b-10)+k)*(dd.grip_q/.04f);')
 old='float r[3];  if constexpr(DYNAMICS)'
 # Body leaf velocities must include the independent scalar gripper motion.
 needle='if constexpr(DYNAMICS){float ip[3]'
 assert needle in src
 src=src.replace(needle,'''if(b==10||b==11){float ax[3]={0,1,0},world[3];qrot(s.q[b],ax,world);for(int k=0;k<3;k++)s.v[b][k]+=world[k]*dd.grip_v;crossadd(s.ab[b],s.om[b],world,2*dd.grip_v);}
  '''+needle)
 needle='__syncwarp();if(output){kin<false>'
 assert needle in src
 src=src.replace(needle,'''__syncwarp();
 if(l==0){
  float h=M(timestep,0),mass=M(body_mass,10)+M(body_mass,11)+M(dof_armature,7)+M(dof_armature,8);
  float kp=M(actuator_gainprm,140),kd=-M(actuator_biasprm,152);
  float f=clampf(kp*(dd.grip_target-dd.grip_q)-kd*dd.grip_v,M(actuator_forcerange,28),M(actuator_forcerange,29));
  dd.grip_v+=h*f/(mass+h*kd);dd.grip_q+=h*dd.grip_v;
  if(dd.grip_q<0){dd.grip_q=0;dd.grip_v=fmaxf(dd.grip_v,0);}
  if(dd.grip_q>.04f){dd.grip_q=.04f;dd.grip_v=fminf(dd.grip_v,0);}
 }
 __syncwarp();if(output){kin<false>''')
 return src
