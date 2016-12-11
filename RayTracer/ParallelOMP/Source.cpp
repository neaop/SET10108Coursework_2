#define _USE_MATH_DEFINES
#include <chrono>
#include <fstream>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <thread>
#include <vector>

using namespace std;
using namespace std::chrono;

// A simple random number generator.
double erand48(unsigned short seed[3]) { return (double)rand() / (double)RAND_MAX; }

// Vec structure to hold corrdinate or r, g, b color values
struct Vec {
  double x, y, z;

  // Vec constructor.
  Vec(double x_ = 0, double y_ = 0, double z_ = 0) {
    x = x_;
    y = y_;
    z = z_;
  }

  // Vec methods.
  Vec operator+(const Vec &b) const { return Vec(x + b.x, y + b.y, z + b.z); }
  Vec operator-(const Vec &b) const { return Vec(x - b.x, y - b.y, z - b.z); }
  Vec operator*(double b) const { return Vec(x * b, y * b, z * b); }
  Vec mult(const Vec &b) const { return Vec(x * b.x, y * b.y, z * b.z); }
  Vec &norm() { return *this = *this * (1 / sqrt(x * x + y * y + z * z)); }
  double dot(const Vec &b) const { return x * b.x + y * b.y + z * b.z; }
  Vec operator%(Vec &b) { return Vec(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x); }
};

// A line with an origin and direction.
struct Ray {
  Vec origin, direction;
  Ray(Vec origin_, Vec direction_) : origin(origin_), direction(direction_) {}
};

// Sphere material types.
enum reflection_type { DIFFUSE, SPECULAR, REFRACTIVE };

// Sphere structure - takes a radius, position and colour.
struct Sphere {
  double radius;
  Vec position, emission, color;
  reflection_type reflection;

  // Sphere constructor.
  Sphere(double radius_, Vec position_, Vec emission_, Vec color_, reflection_type reflection_)
      : radius(radius_), position(position_), emission(emission_), color(color_), reflection(reflection_) {}

  // Returns distance of a ray intersection - 0 on a miss.
  double intersect(const Ray &ray) const {
    // Solve: t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0.
    Vec op = position - ray.origin;
    double t;
    double eps = 1e-4;
    double b = op.dot(ray.direction);
    double det = b * b - op.dot(op) + radius * radius;

    if (det < 0) {
      return 0;
    } else {
      det = sqrt(det);
    }
    return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
  }
};

// Scene to be rendered - made entierly of spheres.
Sphere spheres[] = {
    // Radius, position, emission, color, material.
    Sphere(1e5, Vec(1e5 + 1, 40.8, 81.6), Vec(), Vec(.75, .25, .25), DIFFUSE),   // Left
    Sphere(1e5, Vec(-1e5 + 99, 40.8, 81.6), Vec(), Vec(.25, .25, .75), DIFFUSE), // Rght
    Sphere(1e5, Vec(50, 40.8, 1e5), Vec(), Vec(.75, .75, .75), DIFFUSE),         // Back
    Sphere(1e5, Vec(50, 40.8, -1e5 + 170), Vec(), Vec(), DIFFUSE),               // Frnt
    Sphere(1e5, Vec(50, 1e5, 81.6), Vec(), Vec(.75, .75, .75), DIFFUSE),         // Botm
    Sphere(1e5, Vec(50, -1e5 + 81.6, 81.6), Vec(), Vec(.75, .75, .75), DIFFUSE), // Top
    Sphere(16.5, Vec(27, 16.5, 47), Vec(), Vec(1, 1, 1) * .999, SPECULAR),       // Mirr
    Sphere(16.5, Vec(73, 16.5, 78), Vec(), Vec(1, 1, 1) * .999, REFRACTIVE),     // Glas
    Sphere(600, Vec(50, 681.6 - .27, 81.6), Vec(12, 12, 12), Vec(), DIFFUSE)     // Lite
};

// Clamp unbounded colours to be within scale.
inline double clamp(double x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

// Converts doubles to ints within pixel color scale (255).
inline int toInt(double x) { return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }

// Intersect a ray with the scene - return true if it hits anything.
inline bool intersect(const Ray &ray, double &t, int &id) {
  double n = sizeof(spheres) / sizeof(Sphere);
  double inf = t = 1e20;
  double d;

  for (int i = int(n); i--;)
    if ((d = spheres[i].intersect(ray)) && d < t) {
      t = d;
      id = i;
    }

  return t < inf;
}

// Computes the radiance estimate along a ray.
Vec radiance(const Ray &r, int d, unsigned short *Xi) {
  Ray ray = r;
  int depth = d;
  double t;        // Distance to intersection
  int id = 0;      // ID of intersected object
  Vec cl(0, 0, 0); // Accumulated color
  Vec cf(1, 1, 1); // Accumulated reflectance

  while (1) {
    // If ray misses - return black.
    if (!intersect(ray, t, id)) {
      return cl;
    }

    const Sphere &obj = spheres[id];                                       // Object hit by ray.
    Vec x = ray.origin + ray.direction * t, n = (x - obj.position).norm(); // Ray intersection point.
    Vec nl = n.dot(ray.direction) < 0 ? n : n * -1;                        // Properly oriented surface normal.
    Vec f = obj.color;                                                     // Object color.
    double p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z;       // Max reflection color.
    cl = cl + cf.mult(obj.emission);

    // Russian roulette - 5 times.
    if (++depth > 5) {
      if (erand48(Xi) < p) {
        f = f * (1 / p);
      } else {
        return cl; // R.R.
      }
    }

    cf = cf.mult(f);

    // If object has a DIFFUSE reflection (not shiny)
    if (obj.reflection == DIFFUSE) {
      double r1 = 2 * M_PI * erand48(Xi);                                        // Angle
      double r2 = erand48(Xi), r2s = sqrt(r2);                                   // Distance from center.
      Vec w = nl;                                                                // Normal.
      Vec u = ((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1)) % w).norm();                // Perpendicular to w.
      Vec v = w % u;                                                             // Perpendicular to u and w.
      Vec d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).norm(); // Random reflection ray.
      // return obj.e + f.mult(radiance(Ray(x,d),depth,Xi));
      ray = Ray(x, d); //
      continue;

      // If object has a SPECULAR reflection.
    } else if (obj.reflection == SPECULAR) {
      // return obj.e + f.mult(radiance(Ray(x,r.d-n*2*n.dot(r.d)),depth,Xi));
      ray = Ray(x, ray.direction - n * 2 * n.dot(ray.direction));
      continue;
    }

    Ray reflRay(x, ray.direction - n * 2 * n.dot(ray.direction)); // Ideal dielectric REFRACTION
    bool into = n.dot(nl) > 0;                                    // Ray from outside going in?
    double nc = 1, nt = 1.5;
    double nnt = into ? nc / nt : nt / nc;
    double ddn = ray.direction.dot(nl);
    double cos2t;

    if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) < 0) { // Total internal reflection
                                                         // return obj.e + f.mult(radiance(reflRay,depth,Xi));
      ray = reflRay;
      continue;
    }

    Vec tdir = (ray.direction * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)))).norm();
    double a = nt - nc, b = nt + nc, R0 = a * a / (b * b), c = 1 - (into ? -ddn : tdir.dot(n));
    double Re = R0 + (1 - R0) * c * c * c * c * c;
    double Tr = 1 - Re, P = .25 + .5 * Re;
    double RP = Re / P, TP = Tr / (1 - P);
    // return obj.e + f.mult(erand48(Xi)<P ?
    //		radiance(reflRay, depth,Xi) * RP:
    //		radiance(Ray(x, tdir), depth, Xi) * TP);
    if (erand48(Xi) < P) {
      cf = cf * RP;
      ray = reflRay;
    } else {
      cf = cf * TP;
      ray = Ray(x, tdir);
    }
    continue;
  }
}

// Execute ray tracing.
void execute(int w, int h, int samp, string time_stamp) {
  int width = w, height = h;                                    // Image dimensions.
  int samples = samp;                                           // Number of samples.
  Ray camera(Vec(50, 52, 295.6), Vec(0, -0.042612, -1).norm()); // Camera position and direction.
  Vec cam_x = Vec(width * .5135 / height);                      // X direction increment.
  Vec cam_y = (cam_x % camera.direction).norm() * .5135;        // Y direction increment.
  Vec color_sample;                                             // Colour samples.
  vector<Vec> pixel_colors;                                     // Vector of color values.
  pixel_colors.resize(width * height);                         // The image being rendered.

#pragma omp parallel for num_threads(16) schedule(static) private(color_sample)
  // Loop over image rows.
  for (int y = 0; y < height; y++) {
    unsigned short Xi[3] = {0, 0, y * y * y};
    // Loop over columns.
    for (unsigned short x = 0; x < width; x++) {
      // 2x2 subpixel rows.
      for (int sy = 0, i = (height - y - 1) * width + x; sy < 2; sy++) {
        // 2x2 subpixel cols.
        for (int sx = 0; sx < 2; sx++, color_sample = Vec()) {
          // For number of samples.
          for (int s = 0; s < samples; s++) {
            double r1 = 2 * erand48(Xi);
            double dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
            double r2 = 2 * erand48(Xi);
            double dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
            // Compute ray direction.
            Vec cam_direction = cam_x * (((sx + .5 + dx) / 2 + x) / width - .5) +
                                cam_y * (((sy + .5 + dy) / 2 + y) / height - .5) + camera.direction;
            // Clamp pixel color values.
            color_sample =
                color_sample +
                radiance(Ray(camera.origin + cam_direction * 140, cam_direction.norm()), 0, Xi) * (1. / samples);
          }
          // Camera rays are pushed forward to start in interior.
          pixel_colors[i] =
              pixel_colors[i] + Vec(clamp(color_sample.x), clamp(color_sample.y), clamp(color_sample.z)) * .25;
        }
      }
    }
  }

  // Print pixels into .ppm file.
  FILE *f = fopen(time_stamp.c_str(), "w");
  fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
  for (int i = 0; i < width * height; i++) {
    fprintf(f, "%d %d %d ", toInt(pixel_colors[i].x), toInt(pixel_colors[i].y), toInt(pixel_colors[i].z));
  }
  fclose(f);
}

int main(int argc, char *argv[]) {
  string samp_no_str(argv[1]);
  // Get current time for file timestamp.
  auto time_stamp = to_string(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());
  // Create file for iteration timings.
  ofstream data("./Data/ParallelOMP/parallelOMP_16S_" + samp_no_str + "SPP_" + time_stamp + ".csv", ofstream::out);
  // Add number of cores to file.
  data << "Cores, 4" << endl;

  // Loop for 100 itterations.
  for (int iteration = 0; iteration < 100; ++iteration) {
    // Output current itteration.
    cout << "Iteration: " << iteration << endl;
    // Get start time.
    auto start_time = system_clock::now();
    // Cacluate number of samples per pixel.
    int samps = argc == 2 ? atoi(argv[1]) / 4 : 1;
    // Execute ray trace.
    execute(512, 512, samps, time_stamp);
    // Get end time.
    auto end_time = system_clock::now();
    // Calculate total time taken.
    auto total_time = duration_cast<milliseconds>(end_time - start_time).count();
    // Output time taken to file.
    data << iteration << "," << total_time << endl;
  }

  // File clean up.
  data.flush();
  data.close();

  return 0;
}