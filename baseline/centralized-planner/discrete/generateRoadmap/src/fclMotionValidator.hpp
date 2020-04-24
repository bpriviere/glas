#pragma once

class fclMotionValidator
  : public ompl::base::MotionValidator
{
public:
  fclMotionValidator(
    ompl::base::SpaceInformationPtr si,
    fcl::CollisionGeometry* environment,
    fcl::CollisionGeometry* robot,
    fcl::Transform3f env_tf)
    : MotionValidator(si)
    , m_environment(environment)
    , m_robot(robot)
    , m_env_tf(env_tf)
  {
  }

  bool checkMotion(const ompl::base::State* s1, const ompl::base::State* s2) const
  {
    if (m_environment && m_robot) {
      const ompl::base::RealVectorStateSpace::StateType* typedS1 = s1->as<ompl::base::RealVectorStateSpace::StateType>();
      const ompl::base::RealVectorStateSpace::StateType* typedS2 = s2->as<ompl::base::RealVectorStateSpace::StateType>();

      fcl::Transform3f robot_tf_beg(fcl::Vec3f((*typedS1)[0], (*typedS1)[1], (*typedS1)[2]));
      fcl::Transform3f robot_tf_end(fcl::Vec3f((*typedS2)[0], (*typedS2)[1], (*typedS2)[2]));

      fcl::ContinuousCollisionRequest request;
      fcl::ContinuousCollisionResult result;
      fcl::continuousCollide(m_environment, m_env_tf, m_env_tf,
                             m_robot, robot_tf_beg, robot_tf_end,
                             request, result);
      return !result.is_collide;
    }
    return true;
  }

  bool checkMotion(const ompl::base::State* s1, const ompl::base::State* s2, std::pair<ompl::base::State*, double>& lastValid) const
  {
    if (m_environment && m_robot) {
      const ompl::base::RealVectorStateSpace::StateType* typedS1 = s1->as<ompl::base::RealVectorStateSpace::StateType>();
      const ompl::base::RealVectorStateSpace::StateType* typedS2 = s2->as<ompl::base::RealVectorStateSpace::StateType>();

      fcl::Transform3f robot_tf_beg(fcl::Vec3f((*typedS1)[0], (*typedS1)[1], (*typedS1)[2]));
      fcl::Transform3f robot_tf_end(fcl::Vec3f((*typedS2)[0], (*typedS2)[1], (*typedS2)[2]));

      fcl::ContinuousCollisionRequest request;
      fcl::ContinuousCollisionResult result;
      fcl::continuousCollide(m_environment, m_env_tf, m_env_tf,
                             m_robot, robot_tf_beg, robot_tf_end,
                             request, result);
      // base::ScopedState<base::RealVectorStateSpace> contactPoint(space);
      ompl::base::State* contactPoint = si_->allocState();
      ompl::base::RealVectorStateSpace::StateType* contactPointTyped = contactPoint->as<ompl::base::RealVectorStateSpace::StateType>();
      for (size_t i = 0; i < 3; ++i) {
        (*contactPointTyped)[i] = result.contact_tf2.getTranslation()[i];
      }
      lastValid.first = contactPoint;
      lastValid.second = result.time_of_contact;
      return !result.is_collide;
    }
    return true;
  }

private:
  fcl::CollisionGeometry* m_environment;
  fcl::CollisionGeometry* m_robot;
  fcl::Transform3f m_env_tf;
};
