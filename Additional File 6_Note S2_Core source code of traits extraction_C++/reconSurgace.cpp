#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <pcl/filters/voxel_grid.h>
using namespace std;

double countPatchArea(pcl::PointXYZ p0, pcl::PointXYZ p1, pcl::PointXYZ p2) {
	double area = -1;
	double side[3]; //存储三条边的长度
	side[0] = sqrt(pow(p0.x - p1.x, 2) + pow(p0.y - p1.y, 2) + pow(p0.z - p1.z, 2));
	side[1] = sqrt(pow(p0.x - p2.x, 2) + pow(p0.y - p2.y, 2) + pow(p0.z - p2.z, 2));
	side[2] = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));

	//海伦公式 s=sqrt(p*(p-a)(p-b)(p-c));
	double p = (side[0] + side[1] + side[2]) / 2;
	if (p*(p - side[0])*(p - side[1])*(p - side[2]) < 0) {
		return 0;
	}
	area = sqrt(p*(p - side[0])*(p - side[1])*(p - side[2]));
	return area;
}

double countMeshArea(pcl::PolygonMesh triangles, pcl::PointCloud<pcl::PointXYZ> verticles) {
	double meshArea = 0;
	for (int i = 0; i < triangles.polygons.size(); i++) {
		pcl::PointXYZ p0 = pcl::PointXYZ(verticles[triangles.polygons[i].vertices[0]].x, verticles[triangles.polygons[i].vertices[0]].y, verticles[triangles.polygons[i].vertices[0]].z);
		pcl::PointXYZ p1 = pcl::PointXYZ(verticles[triangles.polygons[i].vertices[1]].x, verticles[triangles.polygons[i].vertices[1]].y, verticles[triangles.polygons[i].vertices[1]].z);
		pcl::PointXYZ p2 = pcl::PointXYZ(verticles[triangles.polygons[i].vertices[2]].x, verticles[triangles.polygons[i].vertices[2]].y, verticles[triangles.polygons[i].vertices[2]].z);
		meshArea += countPatchArea(p0, p1, p2);
	}
	return meshArea;
}

double reconSurface(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float resolution_Z, string surface)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::VoxelGrid<pcl::PointXYZ> sor;
	sor.setInputCloud(cloud);
	sor.setLeafSize(resolution_Z, resolution_Z, resolution_Z);
	sor.filter(*cloud_filtered);


	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(cloud_filtered);
	n.setInputCloud(cloud_filtered);
	n.setSearchMethod(tree);
	n.setKSearch(20);
	n.compute(*normals);
	

	// Concatenate the XYZ and normal fields （将点云和法线放在一起）
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
	pcl::concatenateFields(*cloud_filtered, *normals, *cloud_with_normals);


	// Create search tree*
	pcl::search::KdTree<pcl::PointNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointNormal>);
	tree2->setInputCloud(cloud_with_normals);

	// Initialize objects （初始化对象）
	pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
	pcl::PolygonMesh triangles;

	//设置参数
	gp3.setSearchRadius(resolution_Z*5);
	gp3.setMu(2.5);
	gp3.setMaximumNearestNeighbors(25); //设置搜索的最近邻点的最大数量
	gp3.setMaximumSurfaceAngle(M_PI / 2); // 45 degrees 最大平面角
	gp3.setMinimumAngle(M_PI / 18); // 10 degrees 每个三角的最大角度
	gp3.setMaximumAngle(3 * M_PI / 4); // 120 degrees
	gp3.setNormalConsistency(false); //若法向量一致，设为true

	// 设置搜索方法和输入点云
	gp3.setInputCloud(cloud_with_normals);
	gp3.setSearchMethod(tree2);

	//执行重构，结果保存在triangles中
	gp3.reconstruct(triangles);
	
	double surface_area = countMeshArea(triangles, *cloud_filtered);
	//保存网格图
	pcl::io::savePLYFileBinary(surface, triangles); //这句可以没有
	return surface_area;
}