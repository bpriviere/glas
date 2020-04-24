#pragma once

class fclStateValidityChecker
  : public ompl::base::StateValidityChecker
{
public:
  fclStateValidityChecker(
    ompl::base::SpaceInformationPtr si,
    fcl::CollisionGeometry* environment,
    fcl::CollisionGeometry* robot,
    fcl::Transform3f env_tf)
    : StateValidityChecker(si)
    , m_environment(environment)
    , m_robot(robot)
    , m_env_tf(env_tf)
  {
  }

  bool isValid (const ompl::base::State* state) const
  {
    if (m_environment && m_robot) {
      const ompl::base::RealVectorStateSpace::StateType* typedState = state->as<ompl::base::RealVectorStateSpace::StateType>();

      fcl::Transform3f robot_tf(fcl::Vec3f((*typedState)[0], (*typedState)[1], (*typedState)[2]));
      fcl::CollisionRequest request;
      fcl::CollisionResult result;
      fcl::collide(m_environment, m_env_tf, m_robot, robot_tf, request, result);
      return !result.isCollision();
    }
    return true;
  }

private:
  fcl::CollisionGeometry* m_environment;
  fcl::CollisionGeometry* m_robot;
  fcl::Transform3f m_env_tf;
};
