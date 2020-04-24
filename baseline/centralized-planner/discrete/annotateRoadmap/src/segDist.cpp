#include <vector>
#include <Eigen/Core>
#include <iostream>

#define SMALL_NUM 0.00000001
#define ABS(x) ((x) >= 0 ? (x) : -(x))   //  absolute value

typedef Eigen::Matrix<float, 3, 1, Eigen::DontAlign> vec3f_t;

// dist3D_Segment_to_Segment(): get the 3D minimum distance between 2 segments
//    Input:  two 3D line segments S1 and S2
//    Return: the shortest distance between S1 and S2
float seg2seg_min_distance(const vec3f_t& a0,const vec3f_t& a1, const vec3f_t& b0, const vec3f_t& b1, vec3f_t& sol1, vec3f_t& sol2)
{
    vec3f_t  u = a1 - a0; //dir for segment 1
    vec3f_t  v = b1 - b0; //dir for segment 2
    vec3f_t  w = a0 - b0; 

    float    a = u.dot(u);         // always >= 0
    float    b = u.dot(v);
    float    c = v.dot(v);         // always >= 0
    float    d = u.dot(w);
    float    e = v.dot(w);
    float    D = a*c - b*b;        // always >= 0
    float    sc, sN, sD = D;       // sc = sN / sD, default sD = D >= 0
    float    tc, tN, tD = D;       // tc = tN / tD, default tD = D >= 0

    // compute the line parameters of the two closest points
    if (D < SMALL_NUM) { // the lines are almost parallel
        sN = 0.0;         // force using point P0 on segment S1
        sD = 1.0;         // to prevent possible division by 0.0 later
        tN = e;
        tD = c;
    }
    else {                 // get the closest points on the infinite lines
        sN = (b*e - c*d);
        tN = (a*e - b*d);
        if (sN < 0.0) {        // sc < 0 => the s=0 edge is visible
            sN = 0.0;
            tN = e;
            tD = c;
        }
        else if (sN > sD) {  // sc > 1  => the s=1 edge is visible
            sN = sD;
            tN = e + b;
            tD = c;
        }
    }

    if (tN < 0.0) {            // tc < 0 => the t=0 edge is visible
        tN = 0.0;
        // recompute sc for this edge
        if (-d < 0.0)
            sN = 0.0;
        else if (-d > a)
            sN = sD;
        else {
            sN = -d;
            sD = a;
        }
    }
    else if (tN > tD) {      // tc > 1  => the t=1 edge is visible
        tN = tD;
        // recompute sc for this edge
        if ((-d + b) < 0.0)
            sN = 0;
        else if ((-d + b) > a)
            sN = sD;
        else {
            sN = (-d +  b);
            sD = a;
        }
    }
    // finally do the division to get sc and tc
    sc = (ABS(sN) < SMALL_NUM ? 0.0 : sN / sD);
    tc = (ABS(tN) < SMALL_NUM ? 0.0 : tN / tD);

    // get the difference of the two closest points
    vec3f_t dP = w + (sc * u) - (tc * v);  // =  S1(sc) - S2(tc)
    sol1 = sc*u + a0;
    sol2 = tc*v + b0;

    return dP.norm();   // return the closest distance
}


int main(int argc, char** argv) {

    vec3f_t s1_p0(0,0,0);
    vec3f_t s1_p1(0,0,1);
    vec3f_t s2_p0(0,0,0);
    vec3f_t s2_p1(0,0,1);

    vec3f_t sol1,sol2;

    std::cout << "Distance: " << seg2seg_min_distance(s1_p0,s1_p1,s2_p0,s2_p1,sol1,sol2) << std::endl;
    std::cout << "Seg 1 sol: " << sol1.transpose() << std::endl;
    std::cout << "Seg 2 sol: " << sol2.transpose() << std::endl;

}