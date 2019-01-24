/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using namespace std;

using std::string;
using std::vector;
using std::normal_distribution;
using std::numeric_limits;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles

  for(int i = 0; i < num_particles; ++i) {
    Particle p;
    p.id = i;
    normal_distribution<double> dist_x(x, 0.3);
    normal_distribution<double> dist_y(y, 0.3);
    normal_distribution<double> dist_theta(theta, 0.01);
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1;
    particles.push_back(p);
  }
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  for(int i = 0; i < particles.size(); i++) {

    double x = velocity/yaw_rate * (sin(particles.at(i).theta + yaw_rate * delta_t) - sin(particles.at(i).theta));
    double y = velocity/yaw_rate * (cos(particles.at(i).theta) - cos(particles.at(i).theta + yaw_rate*delta_t));
    double theta = particles.at(i).theta + yaw_rate * delta_t;

    normal_distribution<double> dist_x(x, 0.3);
    normal_distribution<double> dist_y(y, 0.3);
    normal_distribution<double> dist_theta(theta, 0.01);

    particles.at(i).x = dist_x(gen);
    particles.at(i).y = dist_y(gen);
    particles.at(i).theta = dist_theta(gen);

  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  for (int i = 0; i < observations.size(); i++) {

    double min_dist = INFINITY;
    int index = -1;

    for (int j = 0; j < predicted.size(); j++) {
      double current_dist = sqrt(pow(observations.at(i).x - predicted.at(j).x,2) + pow(observations.at(i).y - predicted.at(j).y,2));

      if(min_dist > current_dist){
        min_dist = current_dist;
        index = j;
      }
    }

    observations.at(i).id = index;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  for(int i = 0; i < particles.size(); i++) {

    Particle p = particles.at(i);

    vector<LandmarkObs> predictions;

    for(int j =0; j < map_landmarks.landmark_list.size(); j++) {
      float lm_x = map_landmarks.landmark_list.at(j).x_f;
      float lm_y = map_landmarks.landmark_list.at(j).x_f;
      int lm_id = map_landmarks.landmark_list.at(j).id_i;
      float dist = sqrt(pow((lm_x - p.x),2) + pow((lm_y-p.y),2));
      if(dist < sensor_range) {
        predictions.push_back(LandmarkObs{ lm_id, lm_x, lm_y});
      }

    }

    vector<LandmarkObs> observations_map;
    for(int j = 0; j < observations.size(); j++) {
      LandmarkObs ob;
      ob.x = p.x + (cos(p.theta) * observations.at(j).x) - (sin(p.theta) * observations.at(j).y);
      ob.y = p.y + (sin(p.theta) * observations.at(j).x) + (cos(p.theta) * observations.at(j).y);
      observations_map.push_back(ob);
    }

    dataAssociation(predictions, observations_map);

    p.weight = 1.0;

    for(int j = 0; j < observations_map.size(); j++) {
      LandmarkObs ob = observations_map.at(j);

      LandmarkObs predicted;
      for(int k = 0; k < predictions.size(); k++) {
        if(predictions.at(k).id == ob.id) {
          predicted = predictions.at(k);
        }
      }

      double s_x = std_landmark[0];
      double s_y = std_landmark[1];
      double weight = 1/(2*M_PI*s_x*s_y) * exp(-1*(pow(predicted.x-ob.x,2)/(2*pow(s_x,2))+(pow(predicted.y-ob.y,2)/(2*pow(s_y,2)))));
      p.weight *= weight;
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  vector<Particle> new_particles;

  vector<double> weidhts;
  for (int i = 0; i < particles.size(); i++) {
    weights.push_back(particles.at(i).weight);
  }

  uniform_int_distribution<int> distribution_int(0, particles.size());
  auto index = distribution_int(gen);

  double max_weight = *max_element(weights.begin(), weights.end());

  uniform_real_distribution<double> distribution_real(0.0, max_weight);
  double beta = 0.0;

  for (int i = 0; i < particles.size(); i++) {
    beta += distribution_real(gen) * 2.0;
    while(beta > weights.at(index)) {
      beta -= weidhts.at(index);
      index = (index + 1) % particles.size();
    }
    new_particles.push_back(particles.at(index));
  } 

  particles = new_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}