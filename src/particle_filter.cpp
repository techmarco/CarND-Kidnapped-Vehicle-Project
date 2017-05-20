/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

#define YAW_RATE_LIMIT 1.e-6

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = 5;
  particles.resize(num_particles);
  weights.resize(num_particles);

  default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (auto &particle : particles) {
      particle.x = dist_x(gen);
      particle.y = dist_y(gen);
      particle.theta = dist_theta(gen);
      particle.weight = 1.0;
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  default_random_engine gen;

  for (auto &particle : particles) {
      double x = particle.x;
      double y = particle.y;
      double theta = particle.theta;

      if (yaw_rate > -YAW_RATE_LIMIT && yaw_rate < YAW_RATE_LIMIT) {
          x += velocity * delta_t * cos(theta);
          y += velocity * delta_t * sin(theta);
      } else {
          x += velocity / yaw_rate * ( sin(theta + yaw_rate * delta_t) - sin(theta) );
          y += velocity / yaw_rate * ( cos(theta) - cos(theta + yaw_rate*delta_t) );
      }
      theta += yaw_rate * delta_t;

      normal_distribution<double> dist_x(x, std_pos[0]);
      normal_distribution<double> dist_y(y, std_pos[1]);
      normal_distribution<double> dist_theta(theta, std_pos[2]);

      particle.x = dist_x(gen);
      particle.y = dist_y(gen);
      particle.theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  for (auto &observation: observations) {
    double cur_dist = std::numeric_limits<double>::max();
    double obs_x = observation.x;
    double obs_y = observation.y;
    for (auto predict: predicted) {
      double pred_x = predict.x;
      double pred_y = predict.y;
      double test_dist = dist(obs_x, obs_y, pred_x, pred_y);
      if ( test_dist < cur_dist ) {
        cur_dist = test_dist;
        observation.id = predict.id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

  // covert Map landmarks to observation type
  std::vector<LandmarkObs> pred;
  pred.resize(map_landmarks.landmark_list.size());

  for (int i=0; i<pred.size(); i++) {
    pred[i].x = map_landmarks.landmark_list[i].x_f;
    pred[i].y = map_landmarks.landmark_list[i].y_f;
    pred[i].id = map_landmarks.landmark_list[i].id_i;
  }

  for (int i=0; i<num_particles; i++) {
    // covert current observations to global coordinates
    std::vector<LandmarkObs> global_obs;
    global_obs.resize(observations.size());

    for (int j=0; j<global_obs.size(); j++) {
      double x = observations[j].x;
      double y = observations[j].y;
      double theta = particles[i].theta;

      global_obs[j].x = particles[i].x + x*cos(theta) - y*sin(theta);
      global_obs[j].y = particles[i].y + x*sin(theta) + y*cos(theta);
      global_obs[j].id = -1;
    }

    // perform data association
    dataAssociation(pred, global_obs);

    std::vector<int> associate;
    std::vector<double> sense_x;
    std::vector<double> sense_y;

    int obs_size = observations.size();
    associate.resize(obs_size);
    sense_x.resize(obs_size);
    sense_y.resize(obs_size);

    // update weights
    weights[i] = 1.;
    for (int j=0; j<global_obs.size(); j++) {
      double diffx = pred[global_obs[j].id-1].x - global_obs[j].x;
      double diffy = pred[global_obs[j].id-1].y - global_obs[j].y;
      weights[i] *= 1./(2. * M_PI * std_landmark[0] * std_landmark[1]) * exp( -1.* ( \
                      pow(pred[global_obs[j].id-1].x - global_obs[j].x, 2) / (2. * pow(std_landmark[0], 2)) + \
                      pow(pred[global_obs[j].id-1].y - global_obs[j].y, 2) / (2. * pow(std_landmark[1], 2)) ) );
      associate[j] = global_obs[j].id;
      sense_x[j] = global_obs[j].x;
      sense_y[j] = global_obs[j].y;
    }

    particles[i].weight = weights[i];
    particles[i] = SetAssociations(particles[i], associate, sense_x, sense_y);
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::vector<Particle> resampled;
  resampled.resize(num_particles);

  std::default_random_engine gen;
  std::uniform_int_distribution<int> dist_index(0, num_particles-1);
  std::uniform_real_distribution<double> dist_beta(0, *max_element(weights.begin(), weights.end()));

  int index = dist_index(gen);
  double beta = 0.;

  for (int i=0; i<num_particles; i++) {
    beta += dist_beta(gen);
    while (weights[index] < beta) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resampled[i] = particles[index];
  }

  particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
