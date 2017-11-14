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

using namespace std;

default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    num_particles = 100;
    weights.resize(num_particles, 1.0f);

    // From C5
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for (int i = 0; i < num_particles; i++) {
        Particle p;

        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;
        particles.push_back(p);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    normal_distribution<double> noise_x(0, std_pos[0]);
    normal_distribution<double> noise_y(0, std_pos[1]);
    normal_distribution<double> noise_theta(0, std_pos[2]);

    for (int i = 0; i < num_particles; i++) {

        // avoid divide by 0
        if (fabs(yaw_rate) < 1e-5) {
            particles[i].x += delta_t * velocity * cos(particles[i].theta);
            particles[i].y += delta_t * velocity * sin(particles[i].theta);
        } else {
            // From C7
            double new_theta = particles[i].theta + yaw_rate*delta_t;
            particles[i].x += velocity / yaw_rate * (sin(new_theta) - sin(particles[i].theta));
            particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(new_theta));
            particles[i].theta = new_theta;
        }

        // adding noise
        particles[i].x += noise_x(gen);
        particles[i].y += noise_y(gen);
        particles[i].theta += noise_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    double min_dist, dis;
    for (int i = 0; i < observations.size(); i++) {
        LandmarkObs obs = observations[i];
        min_dist = std::numeric_limits<double>::max();
        int min_j = -1;

        for (int j = 0; j < predicted.size(); j++) {
            LandmarkObs pred = predicted[j];
            dis = dist(obs.x, obs.y, pred.x, pred.y);
            if (dis < min_dist) {
                min_dist = dis;
                min_j = j;
            }
        }
        observations[i].id = min_j;
    }
}

LandmarkObs homogenousTransformation(Particle p, LandmarkObs o) {
    LandmarkObs ret;

    // From C15
    ret.x = p.x + cos(p.theta) * o.x - sin(p.theta) * o.y;
    ret.y = p.y + sin(p.theta) * o.x + cos(p.theta) * o.y;
    ret.id = o.id;
    return ret;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    for (int i = 0; i < particles.size(); i++) {
        Particle p = particles[i];

        vector<LandmarkObs> pred_landmark;

        for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
            LandmarkObs lm_pred;
            lm_pred.x = map_landmarks.landmark_list[j].x_f;
            lm_pred.y = map_landmarks.landmark_list[j].y_f;
            lm_pred.id = map_landmarks.landmark_list[j].id_i;

            if (dist(p.x, p.y, lm_pred.x, lm_pred.y) <= sensor_range) {
                pred_landmark.push_back(lm_pred);
            }
        }

        vector<LandmarkObs> transformed_obs;
        double total_prob = 1.0f;
        for (int j = 0; j < observations.size(); j++) {
            LandmarkObs obsInMap = homogenousTransformation(p, observations[j]);
            transformed_obs.push_back(obsInMap);
        }

        dataAssociation(pred_landmark, transformed_obs);

        for (int j = 0; j < transformed_obs.size(); j++) {
            LandmarkObs obs = transformed_obs[j];
            LandmarkObs lm = pred_landmark[obs.id];

            // From C19
            double sig_x = 0.3;
            double sig_y = 0.3;
            double gauss_norm = 2.0 * M_PI * sig_x * sig_y;
            double exponent = ((obs.x - lm.x)*(obs.x - lm.x))/(2 * sig_x*sig_x) + ((obs.y - lm.y)*(obs.y - lm.y))/(2 * sig_y*sig_y);
            double w = exp(-exponent) / gauss_norm;

            total_prob *= w;
        }
        particles[i].weight = total_prob;
        weights[i] = total_prob;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    std::discrete_distribution<int> d(weights.begin(), weights.end());
    std::vector<Particle> new_particles;

    for (int i = 0; i < num_particles; i++) {
        int ind = d(gen);
        new_particles.push_back(particles[ind]);
    }
    particles = new_particles;
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
