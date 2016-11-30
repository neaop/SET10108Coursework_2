#define _USE_MATH_DEFINES
#include <chrono>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <mpi.h>
#include <exception>
#include <vector>

using namespace std;
using namespace std::chrono;

double erand48(unsigned short seed[3]) {
	return (double)rand() / (double)RAND_MAX;
}

// Structure to hold position of points 
struct Vec {
	double x, y, z;
	Vec(double x_ = 0, double y_ = 0, double z_ = 0) {
		x = x_;
		y = y_;
		z = z_;
	}
	Vec operator+(const Vec &b) const { return Vec(x + b.x, y + b.y, z + b.z);	}
	Vec operator-(const Vec &b) const { return Vec(x - b.x, y - b.y, z - b.z);	}
	Vec operator*(double b) const { return Vec(x * b, y * b, z * b);			}
	Vec mult(const Vec &b) const { return Vec(x * b.x, y * b.y, z * b.z);		}
	Vec &norm() { return *this = *this * (1 / sqrt(x * x + y * y + z * z));		}
	double dot(const Vec &b) const {return x * b.x + y * b.y + z * b.z;			}
	Vec operator%(Vec &b) {
		return Vec(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x);	}
};

// A line with origin and direction.
struct Ray {
	Vec o, d;
	Ray(Vec o_, Vec d_) : o(o_), d(d_) {}
};

// Material types.
enum Refl_t { DIFF, SPEC, REFR };

// Sphere structure - takes a radius, position and colour.
struct Sphere {
	double rad;  // radius
	Vec p, e, c; // position, emission, color
	Refl_t refl; // reflection type (DIFFuse, SPECular, REFRactive)

	Sphere(double rad_, Vec p_, Vec e_, Vec c_, Refl_t refl_)
		: rad(rad_), p(p_), e(e_), c(c_), refl(refl_) {}

	// Returns distance, 0 if nohit.
	double intersect(const Ray &r) const {
		// Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0.
		Vec op = p - r.o;
		double t, eps = 1e-4;
		double b = op.dot(r.d);
		double det = b * b - op.dot(op) + rad * rad;
		if (det < 0) 
			return 0;
		else 
			det = sqrt(det);
		return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
	}
};

// Scene to be rendered - made entierly of spheres.
Sphere spheres[] = {
	// Scene: radius, position, emission, color, material.
	Sphere(1e5, Vec(1e5 + 1, 40.8, 81.6),	Vec(), Vec(.75, .25, .25),	DIFF), // Left
	Sphere(1e5, Vec(-1e5 + 99, 40.8, 81.6), Vec(), Vec(.25, .25, .75),	DIFF), // Rght
	Sphere(1e5, Vec(50, 40.8, 1e5),			Vec(), Vec(.75, .75, .75),	DIFF), // Back
	Sphere(1e5, Vec(50, 40.8, -1e5 + 170),	Vec(), Vec(),				DIFF), // Frnt
	Sphere(1e5, Vec(50, 1e5, 81.6),			Vec(), Vec(.75, .75, .75),	DIFF), // Botm
	Sphere(1e5, Vec(50, -1e5 + 81.6, 81.6), Vec(), Vec(.75, .75, .75),	DIFF), // Top
	Sphere(16.5, Vec(27, 16.5, 47),			Vec(), Vec(1, 1, 1)*.999,	SPEC), // Mirr
	Sphere(16.5, Vec(73, 16.5, 78),			Vec(), Vec(1, 1, 1)*.999,	REFR), // Glas
	Sphere(600, Vec(50, 681.6 - .27, 81.6), Vec(12, 12, 12),	Vec(),	DIFF)  // Lite
};

// Clamp unbounded colour to be between 0 - 255.
inline double clamp(double x) { 
	return x < 0 ? 0 : x > 1 ? 1 : x; 
}

// Converts doubles to ints to be saved into .ppm file.
inline int toInt(double x) { 
	return int(pow(clamp(x), 1 / 2.2) * 255 + .5); 
}

// Intersect a ray with the scene.
inline bool intersect(const Ray &r, double &t, int &id) {

	double n = sizeof(spheres) / sizeof(Sphere);
	double d;
	double inf = t = 1e20;
	for (int i = int(n); i--;)
		if ((d = spheres[i].intersect(r)) && d < t) {
			t = d;
			id = i;
		}
	return t < inf;
}

// Computes the radiance estimate along a ray.
Vec radiance(const Ray &r_, int depth_, unsigned short *Xi) {

	double t;	// distance to intersection
	int id = 0;	// id of intersected object
	Ray r = r_;
	int depth = depth_;
	Vec cl(0, 0, 0);	// accumulated color
	Vec cf(1, 1, 1);	// accumulated reflectance

	while (1) {
		if (!intersect(r, t, id))
			return cl;	// if miss, return black
		const Sphere &obj = spheres[id];	// the hit object
		Vec x = r.o + r.d * t, n = (x - obj.p).norm(), nl = n.dot(r.d) < 0 ? n : n * -1, f = obj.c;
		double p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z;	// max refl
		cl = cl + cf.mult(obj.e);
		if (++depth > 5)
			if (erand48(Xi) < p)
				f = f * (1 / p);
			else
				return cl;	// R.R.
		cf = cf.mult(f);
		if (obj.refl == DIFF) {	// Ideal DIFFUSE reflection
			double r1 = 2 * M_PI * erand48(Xi), r2 = erand48(Xi), r2s = sqrt(r2);
			Vec w = nl, u = ((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1)) % w).norm(), v = w % u;
			Vec d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).norm();
			// return obj.e + f.mult(radiance(Ray(x,d),depth,Xi));
			r = Ray(x, d);
			continue;
		}
		else if (obj.refl == SPEC) { // Ideal SPECULAR reflection
									 // return obj.e + f.mult(radiance(Ray(x,r.d-n*2*n.dot(r.d)),depth,Xi));
			r = Ray(x, r.d - n * 2 * n.dot(r.d));
			continue;
		}
		Ray reflRay(x, r.d - n * 2 * n.dot(r.d)); // Ideal dielectric REFRACTION
		bool into = n.dot(nl) > 0;                // Ray from outside going in?
		double nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = r.d.dot(nl), cos2t;
		if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) <
			0) { // Total internal reflection
				 // return obj.e + f.mult(radiance(reflRay,depth,Xi));
			r = reflRay;
			continue;
		}
		Vec tdir =
			(r.d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)))).norm();
		double a = nt - nc, b = nt + nc, R0 = a * a / (b * b), c = 1 - (into ? -ddn : tdir.dot(n));
		double Re = R0 + (1 - R0) * c * c * c * c * c, Tr = 1 - Re, P = .25 + .5 * Re, RP = Re / P, TP = Tr / (1 - P);
		// return obj.e + f.mult(erand48(Xi)<P ?
		//                       radiance(reflRay,    depth,Xi)*RP:
		//                       radiance(Ray(x,tdir),depth,Xi)*TP);
		if (erand48(Xi) < P) {
			cf = cf * RP;
			r = reflRay;
		}
		else {
			cf = cf * TP;
			r = Ray(x, tdir);
		}
		continue;
	}
}

MPI_Datatype createMPIVec() {

	MPI_Datatype VecType;
	MPI_Datatype type[3] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
	int blockLen[3] = { 1,1,1 };
	MPI_Aint disp[3];

	disp[0] = (MPI_Aint)offsetof(struct Vec, x);
	disp[1] = (MPI_Aint)offsetof(struct Vec, y);
	disp[2] = (MPI_Aint)offsetof(struct Vec, z);

	MPI_Type_create_struct(3, blockLen, disp, type, &VecType);
	MPI_Type_commit(&VecType);
	return VecType;
}

void execute(int width, int height, int samples, int my_rank, int num_procs) {

	int w = width, h = height;	// Image dimensions.
	int samps = samples;	// Number of samples.
	
	int chunk = h / num_procs;
	int chunk_end;

	chunk_end = (num_procs - (my_rank)) * chunk;

	Ray cam(Vec(50, 52, 295.6), Vec(0, -0.042612, -1).norm()); // Camera position and direction.
	
	Vec cx = Vec(w * .5135 / h);					// X direction increment.
	Vec cy = (cx % cam.d).norm() * .5135;			// Y direction increment.
	Vec r;											// Colour samples.
	Vec *my_pixels = new Vec[w * chunk];			// The image being rendered.

	MPI_Datatype mpi_vec = createMPIVec();

	//if (my_rank == 0) {
	//	double wut = h / num_procs;
	//	std::cout << "chunk = " << chunk << std::endl;
	//}
	//std::cout << my_rank << " start = " << chunk* my_rank << " end = " << chunk_end << std::endl;

	for (int y = chunk * (num_procs - (my_rank + 1)); y < chunk_end; y++) {	// Loop over image rows.

		unsigned short Xi[3] = { 0, 0, y * y * y };
		
		for (unsigned short x = 0; x < w; x++) { 
			for (int sy = 0, i = (chunk_end - y - 1) * w + x; sy < 2; sy++) {
				for (int sx = 0; sx < 2; sx++, r = Vec()) {		
					for (int s = 0; s < samps; s++) {

						double r1 = 2 * erand48(Xi);
						double dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
						double r2 = 2 * erand48(Xi);
						double dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
						
						Vec d = cx * (((sx + .5 + dx) / 2 + x) / w - .5) +
							cy * (((sy + .5 + dy) / 2 + y) / h - .5) + cam.d; // Compute ray direction
						r = r + radiance(Ray(cam.o + d * 140, d.norm()), 0, Xi) * (1. / samps);
					}	
					// Camera rays are pushed ^^^^^ forward to start in interior.
					my_pixels[i] = my_pixels[i] + Vec(clamp(r.x), clamp(r.y), clamp(r.z)) * .25;
				}
			}
		}
	}
	
	vector<Vec> all_pixels;	// Declare datastructure for all pixels

	if (my_rank == 0) {
		all_pixels.resize(w * h);	// Initialize pixel data structure
		std::cout << "Commencing gather." << std::endl;		
	}
	else {

	}
	// Gather individual processor pixels into proc 0.
	MPI_Gather(&my_pixels[0], chunk*w, mpi_vec, &all_pixels[0], chunk*w, mpi_vec, 0, MPI_COMM_WORLD);

	// Write pixel values to file.
	if (my_rank == 0) {
		std::cout << "Drawing image." << std::endl;
		FILE *f = fopen("image.ppm", "w"); 
		fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);

		/*for (int p = num_procs - 1; p > -1; p--) {
			for (int i = p * chunk*w; i < (p+1) * chunk * w; i++) {
		*/		
		for (size_t i = 0; i < w*h; i++)
		{
		fprintf(f, "%d %d %d ", toInt(all_pixels[i].x), toInt(all_pixels[i].y), toInt(all_pixels[i].z));
			}
		}


}

int main(int argc, char *argv[]) {

	// Initialise MPI.
	int num_procs, my_rank;
	auto result = MPI_Init(nullptr, nullptr);

	if (result != MPI_SUCCESS) {
		MPI_Abort(MPI_COMM_WORLD, result);
		return -1;
	}

	// Get MPI info.
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	std::ofstream data;
	time_point<system_clock> start_time;

	int width = 4096;
	int hight = 4096;

	if (my_rank == 0) {
		// Get current time for timings file timestamp.
		auto time_stamp = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
		// Create timings file.
		std::ofstream data("./Data/parallelMPI_" + std::to_string(time_stamp) + ".csv", std::ofstream::out);
		start_time = system_clock::now();
	}

	int samps = argc == 2 ? atoi(argv[1]) / 4 : 1;

	execute(width,hight,samps, my_rank, num_procs);

	if (my_rank == 0) {
		auto end_time = system_clock::now();
		auto total_time = duration_cast<milliseconds>(end_time - start_time).count();
		data << "," << total_time << std::endl;
		data.flush();
		data.close();
	}

	MPI_Finalize();
	return 0;
}