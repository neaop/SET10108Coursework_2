#define _USE_MATH_DEFINES
#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <set>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
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

MPI_Datatype createMPIVec() {

  MPI_Datatype VecType;
  MPI_Datatype type[3] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
  int blockLen[3] = {1, 1, 1};
  MPI_Aint disp[3];

  disp[0] = (MPI_Aint)offsetof(struct Vec, x);
  disp[1] = (MPI_Aint)offsetof(struct Vec, y);
  disp[2] = (MPI_Aint)offsetof(struct Vec, z);

  MPI_Type_create_struct(3, blockLen, disp, type, &VecType);
  MPI_Type_commit(&VecType);
  return VecType;
}

void execute(int w, int h, int samps, string time_stamp, int my_rank, int num_procs) {

  int width = w, height = h;                                    // Image dimensions.
  int samples = samps;                                          // Number of samples.
  int chunk_size = height / num_procs;                          // Amount of work for each node .
  int node_start = (num_procs - (my_rank + 1)) * chunk_size;    // Node start index.
  int node_end = (num_procs - (my_rank)) * chunk_size;          // Node end index.
  Ray camera(Vec(50, 52, 295.6), Vec(0, -0.042612, -1).norm()); // Camera position and direction.
  Vec cam_x = Vec(width * .5135 / height);                      // X direction increment.
  Vec cam_y = (cam_x % camera.direction).norm() * .5135;        // Y direction increment.
  Vec color_sample;                                             // Colour samples.
  vector<Vec> pixel_colors;                                     // Vector of pixel values
  pixel_colors.reserve(width * chunk_size);

  // Loop over chunk rows.
  for (int y = node_start; y < node_end; y++) {
    unsigned short Xi[3] = {0, 0, y * y * y};
    // Loop over columns.
    for (unsigned short x = 0; x < width; x++) {
      // 2x2 subpixel rows.
      for (int sy = 0, i = (node_end - y - 1) * width + x; sy < 2; sy++) {
        // 2x2 subpixel cols.
        for (int sx = 0; sx < 2; sx++, color_sample = Vec()) {
          // For number of samples.
          for (int s = 0; s < samples; s++) {
            double r1 = 2 * erand48(Xi);
            double dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
            double r2 = 2 * erand48(Xi);
            double dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
            // Compute ray direction
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

  MPI_Datatype mpi_vec = createMPIVec(); // Declare custom MPI object to transfer pixel colors.
  vector<Vec> all_pixels;                // Declare datastructure for all pixels

  if (my_rank == 0) {
    // Initialize pixel data structure
    all_pixels.resize(width * height);
  }

  // Gather individual processor pixels into proc 0.
  MPI_Gather(&pixel_colors[0], chunk_size * width, mpi_vec, &all_pixels[0], chunk_size * width, mpi_vec, 0,
             MPI_COMM_WORLD);

  // Host writes pixel values to file.
  if (my_rank == 0) {
    std::cout << "Drawing image." << std::endl;
    FILE *f = fopen(time_stamp.c_str(), "w");
    fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
    for (size_t i = 0; i < width * height; i++) {
      fprintf(f, "%d %d %d ", toInt(all_pixels[i].x), toInt(all_pixels[i].y), toInt(all_pixels[i].z));
    }
    fclose(f);
  }
}

// Return the number of physical computers being used.
int get_host_num(int my_rank, int num_procs) {

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  if (my_rank != 0) {
    MPI_Send(processor_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
  }

  else {
    set<string> names;
    names.insert(processor_name);
    char temp_name[MPI_MAX_PROCESSOR_NAME];

    for (int i = 1; i < num_procs; ++i) {
      MPI_Recv(&temp_name[0], MPI_MAX_PROCESSOR_NAME, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      names.insert(temp_name);
    }
    return names.size();
  }
}

int main(int argc, char *argv[]) {

  // Exit if no sample number provided.
  if (argc < 2) {
    cout << "Invalid arguments." << endl;
    return -1;
  }

  // Initialise MPI.
  auto result = MPI_Init(&argc, &argv);

  if (result != MPI_SUCCESS) {
    MPI_Abort(MPI_COMM_WORLD, result);
    return -1;
  }

  // Get MPI info.
  int num_procs, my_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  int hosts = get_host_num(my_rank, num_procs);  // Number of physical hosts.
  string time_stamp = "";                        // Timestamp for unique file name.
  time_point<system_clock> start_time;           // Iteration start time.
  vector<long> execution_times;                  // Vector for all iteration times.
  int samps = argc == 2 ? atoi(argv[1]) / 4 : 1; // Number of samples per pixel.

  if (my_rank == 0) {
    // Master node gets current time for file names.
    time_stamp = to_string(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());
  }

  // For 100 iterations.
  for (int iteration = 0; iteration < 100; ++iteration) {
    if (my_rank == 0) {
      // Output current iteration.
      cout << "Iteration: " << iteration << endl;
      // Get start time.
      start_time = system_clock::now();
    }

    // Execute ray trace.
    execute(512, 512, samps, time_stamp, my_rank, num_procs);

    if ((my_rank == 0) && (iteration > 9)) {
      // Get end time.
      auto end_time = system_clock::now();
      // Calculate total time taken.
      auto total_time = duration_cast<milliseconds>(end_time - start_time).count();
      // Push time taken into vector.
      execution_times.push_back(total_time);
    }
  }

  if (my_rank == 0) {
    // Create file name.
    string samp_no_str(argv[1]);
    stringstream ss;
    ss << "./Data/parallel_MPI_" << hosts << "H" << num_procs << "N_" << samp_no_str << "SPP_" << time_stamp << ".csv";
    string file_name = ss.str();
    // Create file writer.
    ofstream data(file_name, ofstream::out);

    // Write execution times to file.
    for (int i = 0; i < execution_times.size(); ++i) {
      data << i << "," << execution_times[i] << endl;
    }

    // File clean up.
    data.flush();
    data.close();
  }

  // MPI clean up.
  MPI_Finalize();

  return 0;
}