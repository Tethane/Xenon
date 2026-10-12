#include <cmath>
#include <cstdio>
#include <stdexcept>
#include "geometry/transform.h"
using namespace xn;

void require(bool b, const char *s) {
	if (!b)
		throw std::runtime_error(s);
}

int main() {
	Mat4x3 matrix;
	matrix.cols[0] = {-2, .3f, .1f};
	matrix.cols[1] = {.5f, 3, .2f};
	matrix.cols[2] = {.2f, .1f, .7f};
	matrix.cols[3] = {10, -3, 2};
	auto transform = AffineTransform::from_matrix(matrix);

	for (Vec3 p : {Vec3(0), Vec3(1, 2, 3), Vec3(-7, .1f, 8)}) {
		auto world = transform.local_to_world.transform_point(p);
		require((transform.world_to_local.transform_point(world) - p).length() < 2e-5,
				"General affine inverse does not restore point");
	}
	Vec3 tangent1{1, 2, 0}, tangent2{0, 1, 2};
	auto normal = normalize(cross(tangent1, tangent2));
	auto world_normal = transform.normal_to_world(normal);
	require(std::abs(dot(world_normal, transform.local_to_world.transform_dir(tangent1))) < 1e-6,
			"Transformed normal is not perpendicular to first tangent");
	require(std::abs(dot(world_normal, transform.local_to_world.transform_dir(tangent2))) < 1e-6,
			"Transformed normal is not perpendicular to second tangent");
	Ray world{{1, 2, 3}, normalize(Vec3(2, 3, 4)), 0, 100};
	auto local = transform.to_local(world);
	require((transform.local_to_world.transform_point(local.at(3.4f)) - world.at(3.4f)).length() <
				1e-5,
			"Instance transform changes ray parameter");
	bool failed = false;
	matrix.cols[1] = matrix.cols[0];

	try {
		AffineTransform::from_matrix(matrix);
	} catch (const std::runtime_error &) {
		failed = true;
	}
	require(failed, "Singular affine instance accepted");
	std::puts("General affine inverse, normals, ray parameter and singular validation passed");
}
