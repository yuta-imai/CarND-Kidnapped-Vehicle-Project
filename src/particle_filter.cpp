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

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for(int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
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

  //We can prepare constant noise for every particle here.
  std::normal_distribution<double> noise_x(0, std_pos[0]);
  std::normal_distribution<double> noise_y(0, std_pos[1]);
  std::normal_distribution<double> noise_theta(0, std_pos[2]);

  for(Particle& p: particles){

    // add measurements to each particle
    if( fabs(yaw_rate) < 0.0001){ 
      p.x += velocity * delta_t * cos(p.theta);
      p.y += velocity * delta_t * sin(p.theta);

    } else{ //Avoiding dividing velocity by too small yaw_rate
      p.x += velocity / yaw_rate * ( sin( p.theta + yaw_rate*delta_t ) - sin(p.theta) );
      p.y += velocity / yaw_rate * ( cos( p.theta ) - cos( p.theta + yaw_rate*delta_t ) );
      p.theta += yaw_rate * delta_t;
    }

    // predicted particles with added sensor noise
    p.x += noise_x(gen);
    p.y += noise_y(gen);
    p.theta += noise_theta(gen);
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

  for(LandmarkObs& obs: observations) {

    double min_dist = std::numeric_limits<float>::max();

    for(const LandmarkObs& pr: predicted) {

      double current_dist = dist(obs.x,obs.y,pr.x,pr.y);
      if(min_dist > current_dist) {
        min_dist = current_dist;
        obs.id = pr.id;
      }
    }
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


  for(auto& p: particles) {

    p.weight = 1.0;

    // Extract landmarks in sensor_range
    vector<LandmarkObs> predictions;
    for(auto& lm: map_landmarks.landmark_list) {
      double distance = dist(p.x,p.y,lm.x_f,lm.y_f);
      if(distance < sensor_range) {
        predictions.push_back(LandmarkObs{lm.id_i,lm.x_f,lm.y_f});
      }
    }

    // Convert observation from vehicle coordinates to map coordinates.
    vector<LandmarkObs> observations_map;
    double cos_theta = cos(p.theta);
    double sin_theta = sin(p.theta);
    for(const LandmarkObs& obs: observations) {
      LandmarkObs obs_map;
      obs_map.x = p.x + cos_theta * obs.x - sin_theta * obs.y;
      obs_map.y = p.y + sin_theta * obs.x + cos_theta * obs.y;
      observations_map.push_back(obs_map);
    }

    // Assosiate landmarks to overvations.
    // What's interesting here is, in this simulator app, we have exact same number of observations with landmarks in range.
    // In more real case, maybe we have to work harder in dataAssosiation?
    dataAssociation(predictions, observations_map);

    // Calculate and accumulate weight
    for(const LandmarkObs& obs_map: observations_map) {
      Map::single_landmark_s landmark = map_landmarks.landmark_list.at(obs_map.id-1);
      double x_term = pow(obs_map.x - landmark.x_f, 2) / (2 * pow(std_landmark[0], 2));
      double y_term = pow(obs_map.y - landmark.y_f, 2) / (2 * pow(std_landmark[1], 2));
      double w = exp(-(x_term + y_term)) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
      p.weight *=  w;
    }

    weights.push_back(p.weight);
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

  //We use resampling wheel in the lecture.
  uniform_int_distribution<int> uniintdist(0, num_particles-1);
  auto index = uniintdist(gen);

  double max_weight = *max_element(weights.begin(), weights.end());

  uniform_real_distribution<double> unirealdist(0.0, 1.0);

  double beta = 0.0;

  for (int i = 0; i < num_particles; i++) {
    beta += unirealdist(gen) * max_weight * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;
  weights.clear();
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