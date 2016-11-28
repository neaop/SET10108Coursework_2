#define _USE_MATH_DEFINES
#include <chrono>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

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
	Vec operator+(const Vec &b) const { return Vec(x + b.x, y + b.y, z + b.z); }
	Vec operator-(const Vec &b) const { return Vec(x - b.x, y - b.y, z - b.z); }
	Vec operator*(double b) const { return Vec(x * b, y * b, z * b); }
	Vec mult(const Vec &b) const { return Vec(x * b.x, y * b.y, z * b.z); }
	Vec &norm() { return *this = *this * (1 / sqrt(x * x + y * y + z * z)); }
	double dot(const Vec &b) const {
		return x * b.x + y * b.y + z * b.z;
	}
	Vec operator%(Vec &b) {
		return Vec(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x);
	}
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
		if (det < 0) {
			return 0;
		}
		else {
			det = sqrt(det);
		}
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
inline double clamp(double x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

// Converts doubles to ints to be saved into .ppm file.
inline int toInt(double x) { return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }

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
	double t;   // distance to intersection
	int id = 0; // id of intersected object
	Ray r = r_;
	int depth = depth_;
	Vec cl(0, 0, 0); // accumulated color
	Vec cf(1, 1, 1); // accumulated reflectance
	while (1) {
		if (!intersect(r, t, id))
			return cl;                     // if miss, return black
		const Sphere &obj = spheres[id]; // the hit object
		Vec x = r.o + r.d * t, n = (x - obj.p).norm(), nl = n.dot(r.d) < 0 ? n : n * -1, f = obj.c;
		double p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z; // max refl
		cl = cl + cf.mult(obj.e);
		if (++depth > 5)
			if (erand48(Xi) < p)
				f = f * (1 / p);
			else
				return cl; // R.R.
		cf = cf.mult(f);
		if (obj.refl == DIFF) { // Ideal DIFFUSE reflection
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

void execute(int samples) {
	int w = 512, h = 384;							// Image dimensions.
	int samps = samples;	// Number of samples.
	Ray cam(Vec(50, 52, 295.6), Vec(0, -0.042612, -1).norm()); // Camera position and direction.
	Vec cx = Vec(w * .5135 / h);			// X direction increment.
	Vec cy = (cx % cam.d).norm() * .5135;	// Y direction increment.
	Vec r;									// Colour samples.
	Vec *c = new Vec[w * h];				// The image being rendered.
#pragma omp parallel for num_threads(2) schedule(static) private(r)
	for (int y = 0; y < h; y++) {			// Loop over image rows.
											// Print progress.
		fprintf(stderr, "\rRendering (%d spp) %5.2f%%", samps * 4, 100. * y / (h - 1));
		unsigned short Xi[3] = { 0, 0, y * y * y };
		for (unsigned short x = 0; x < w; x++)	// Loop over columns

			for (int sy = 0, i = (h - y - 1) * w + x; sy < 2; sy++)	// 2x2 subpixel rows
				for (int sx = 0; sx < 2; sx++, r = Vec()) {			// 2x2 subpixel cols
					for (int s = 0; s < samps; s++) {				// For number of samples.
						double r1 = 2 * erand48(Xi), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
						double r2 = 2 * erand48(Xi), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
						// Compute ray direction
						Vec d = cx * (((sx + .5 + dx) / 2 + x) / w - .5) +
							cy * (((sy + .5 + dy) / 2 + y) / h - .5) + cam.d;

						r = r + radiance(Ray(cam.o + d * 140, d.norm()), 0, Xi) * (1. / samps);
					} // Camera rays are pushed ^^^^^ forward to start in interior.
					c[i] = c[i] + Vec(clamp(r.x), clamp(r.y), clamp(r.z)) * .25;
				}
	}
	FILE *f = fopen("image.ppm", "w"); // Write image to PPM file.
	fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
	for (int i = 0; i < w * h; i++)
		fprintf(f, "%d %d %d ", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));

}

int main(int argc, char *argv[]) {
	// Get current time for  timings file timestamp.
	auto time_stamp = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	// Create timings file.
	std::ofstream data("./Data/parallelOMP8S_" + std::to_string(time_stamp) + ".csv", std::ofstream::out);

	// Loop 100 times
	for (int iteration = 0; iteration < 100; ++iteration) {
		// Output current itteration.
		std::cout << "Iteration: " << iteration << std::endl;
		auto start_time = system_clock::now();

		int samps = argc == 2 ? atoi(argv[1]) / 4 : 1;
		execute(samps);

		auto end_time = system_clock::now();
		auto total_time =
			duration_cast<milliseconds>(end_time - start_time).count();
		data << iteration << "," << total_time << std::endl;
	}
	data.flush();
	data.close();
}