#include <io.h>
#include <direct.h>
#include <stdio.h>
#include <stdlib.h> 
#include <opencv2/opencv.hpp>
#include <pcl\point_cloud.h>
#include <pcl/point_types.h>
using namespace std;
using namespace cv;

#define pi 3.1415926

void connectedAreaBox(Mat labelImage, vector<vector<Point2f>>& tribe) {
	if (labelImage.empty() ||
		labelImage.type() != CV_32SC1) {
		return;
	}
	map<float, vector<Point2f>> box;
	for (int i = 0; i < labelImage.rows; i++) {
		const int* ptr_i = labelImage.ptr<int>(i);
		for (int j = 0; j < labelImage.cols; j++) {
			if (ptr_i[j] > 0) {
				box[ptr_i[j]].push_back(Point2i(i, j));
			}
		}
	}
	map<float, vector<Point2f>>::const_iterator box_it = box.begin();
	int i = 0;
	while (box_it != box.end()) {

		tribe.push_back(box_it->second);
		++box_it;
	}
}

vector<Point2f> extractCenterOfGravity(vector<vector<Point2f>> tribe) {
	vector<Point2f> vecCenterPoints;
	vector<vector<Point2f>>::iterator area_i;
	vector<Point2f>::iterator centerPoints_i;
	for (area_i = tribe.begin(); area_i != tribe.end(); area_i++) {
		float c_x = 0, c_y = 0;
		float x_t = 0, y_t = 0;
		for (centerPoints_i = area_i->begin(); centerPoints_i != area_i->end(); centerPoints_i++) {
			x_t += centerPoints_i->x;
			y_t += centerPoints_i->y;
		}
		c_x = x_t / area_i->size();
		c_y = y_t / area_i->size();
		vecCenterPoints.push_back(Point2f(c_x, c_y));
	}
	return vecCenterPoints;
}

void lineRegression(vector<Point3f>& points, double& x0, double& y0, double& m, double& n);
void drawSampleLine(int& x0, int& y0, int& m, int& n, string sampleLine);
float angleBetweenLineAndPlanarXY(float m, float n);
float angleBetweenLines(float m1, float n1, float m2, float n2);

void areaThreshold(Mat& src, int min, int max) {
	Mat label;
	vector<vector<Point2f>> tribe;
	connectedComponents(src, label);
	connectedAreaBox(label, tribe);
	for (int i = 0; i < tribe.size(); i++) {
		if (tribe[i].size() > min && tribe[i].size() < max) {
			for (int j = 0; j < tribe[i].size(); j++) {
				src.at<uchar>(tribe[i][j].x, tribe[i][j].y) = 0;
			}
		}
	}
}

void deleteSmallArea(Mat& src, int min) {
	Mat label;
	vector<vector<Point2f>> tribe;
	connectedComponents(src, label);
	connectedAreaBox(label, tribe);
	for (int i = 0; i < tribe.size(); i++) {
		if (tribe[i].size() < min) {
			for (int j = 0; j < tribe[i].size(); j++) {
				src.at<uchar>(tribe[i][j].x, tribe[i][j].y) = 0;
			}
		}
	}
}

float max_distance(vector<Point2f>& proj) {
	float dist_max = 0;
	for (int i = 0; i < proj.size(); i++) {
		for (int j = i; j < proj.size(); j++) {
			float dist_temp = (proj[i].x - proj[j].x)*(proj[i].x - proj[j].x) + (proj[i].y - proj[j].y)*(proj[i].y - proj[j].y);
			if (dist_temp > dist_max) {
				dist_max = dist_temp;
			}
		}
	}
	return sqrt(dist_max);
}

void getParamFromMat(Size matSize, vector<Point2f>& points, float& major, float& minor, float& thickness_min, float& area, float& area_out, float& perimeter, float& area_perimeter) {
	Mat bin(matSize, CV_8U, Scalar::all(0));
	for (int i = 0; i < points.size(); i++) {
		bin.at<uchar>(int(points[i].x), int(points[i].y)) = 255;
	}
	area = countNonZero(bin);

	Mat out;
	Mat tube;
	bin.copyTo(out);
	threshold(bin, tube, 128, 255, THRESH_BINARY_INV);
	areaThreshold(tube, 0, 100000);
	threshold(tube, tube, 128, 255, THRESH_BINARY_INV);
	out = tube + bin;
	area_out = countNonZero(out);

	vector<vector<Point>> g_vContours;
	vector<Vec4i> g_vHierarchy;
	findContours(out, g_vContours, g_vHierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	if (g_vContours.size() == 1) {
		perimeter = arcLength(g_vContours[0], true);
		area_perimeter = area_out / perimeter;
	}
	else {
		perimeter = 0;
		area_perimeter = 0;
	}

	Mat edge, edge_dilate;
	Canny(bin, edge, 5, 1000);
	Mat element = getStructuringElement(MORPH_DILATE, Size(3, 3));
	dilate(edge, edge_dilate, element);
	Mat label_edge_dilate;
	connectedComponents(edge_dilate, label_edge_dilate, 4);
	vector<vector<Point2f>> tribe_edge_dilate;
	connectedAreaBox(label_edge_dilate, tribe_edge_dilate);

	if (tribe_edge_dilate.size() < 2) {
		major = 0;
		minor = 0;
		thickness_min = 0;
		return;
	}

	vector<vector<Point2f>> tribe_edge;
	for (int i = 0; i < tribe_edge_dilate.size(); i++) {
		vector<Point2f> temp;
		for (int j = 0; j < tribe_edge_dilate[i].size(); j++) {
			if (edge.at<uchar>(tribe_edge_dilate[i][j].x, tribe_edge_dilate[i][j].y)) {
				temp.push_back(Point2f(tribe_edge_dilate[i][j].x, tribe_edge_dilate[i][j].y));
			}
		}
		tribe_edge.push_back(temp);
	}

	int edgeSize_max = 0, seq_max = 0;
	for (int i = 0; i < tribe_edge.size(); i++) {
		if (tribe_edge[i].size() > edgeSize_max) {
			edgeSize_max = tribe_edge[i].size();
			seq_max = i;
		}
	}


	vector<float> wallThickness_vec;
	for (int i = 0; i < tribe_edge.size(); i++) {
		if (i == seq_max) {}
		else {
			for (int j = 0; j < tribe_edge[i].size(); j++) {
				float dist_min = FLT_MAX;
				Point2f correspondingPoint;
				for (int j_outer = 0; j_outer < tribe_edge[seq_max].size(); j_outer++) {
					float temp = (tribe_edge[i][j].x - tribe_edge[seq_max][j_outer].x)*(tribe_edge[i][j].x - tribe_edge[seq_max][j_outer].x) + (tribe_edge[i][j].y - tribe_edge[seq_max][j_outer].y)*(tribe_edge[i][j].y - tribe_edge[seq_max][j_outer].y);
					if (temp < dist_min) {
						dist_min = temp;
						correspondingPoint = tribe_edge[seq_max][j_outer];
					}
				}
				/*line(display, Point(int(tribe_edge[i][j].y), int(tribe_edge[i][j].x)), Point(int(correspondingPoint.y), int(correspondingPoint.x)), Scalar(0, 0, 255), 1);*/
				wallThickness_vec.push_back(sqrt(dist_min));
			}
		}
	}
	thickness_min = FLT_MAX;
	for (int i = 0; i < wallThickness_vec.size(); i++) {
		if (wallThickness_vec[i] > 0 && wallThickness_vec[i] < thickness_min) {
			thickness_min = wallThickness_vec[i];
		}
	}
	if (thickness_min > 25) {
		thickness_min = 0;
	}

	major = 0, minor = FLT_MAX;
	for (int theta = -90; theta < 90; theta++) {
		float k = tan(theta*3.1415926 / 180);
		vector<Point2f> proj;
		for (int i = 0; i < tribe_edge[seq_max].size(); i++) {
			float x = (tribe_edge[seq_max][i].x + k * tribe_edge[seq_max][i].y) / (k*k + 1);
			float y = k * x;
			proj.push_back(Point2f(x, y));
		}
		float axis_theta = max_distance(proj);
		if (axis_theta > major) {
			major = axis_theta;
		}
		else {
			if (axis_theta < minor) {
				minor = axis_theta;
			}
		}
	}
}

void max_min_V_SD(vector<float> a, float& max, float& min, float& ave, float& V, float& SD) {
	float temp_min = FLT_MAX, temp_max = FLT_MIN, average = 0;
	for (int i = 0; i < a.size(); i++) {
		average += a[i];
		if (a[i] < temp_min) {
			temp_min = a[i];
		}
		if (a[i] > temp_max) {
			temp_max = a[i];
		}
	}
	max = temp_max;
	min = temp_min;
	average /= a.size();
	ave = average;
	float sum = 0;
	for (int i = 0; i < a.size(); i++) {
		sum += (a[i] - average)*(a[i] - average);
	}
	V = sum / a.size();
	SD = sqrt(V);
}


void sort(vector<float> a, vector<float>& b) {
	float temp = 0;
	while (a.size()) {
		temp = a.front();
		a.erase(a.begin());
		vector<float>::iterator i = b.begin();
		while (i != b.end()) {
			if (temp > *i) {
				i++;
			}
			else {
				b.insert(i, temp);
				break;
			}
		}
		if (i == b.end()) {
			b.push_back(temp);
		}
	}
}

void _convexHull(Mat bin, double &area_section, double &area_convexHull, double &area_circle, double &ratio_section_to_convexHull, double &ratio_section_to_circle) {

	vector<vector<Point2f>> tribe;
	Mat label;
	connectedComponents(bin, label, 4);
	connectedAreaBox(label, tribe);

	area_section = countNonZero(bin);

	vector<Point> points;
	for (int i = 0; i < bin.rows; i++) {
		for (int j = 0; j < bin.cols; j++) {
			if (bin.at<uchar>(i, j) > 128) {
				points.push_back(Point(j, i));
			}
		}
	}

	vector<int> hull;
	convexHull(Mat(points), hull, true);

	int hullcount = (int)hull.size();
	Point point0 = points[hull[hullcount - 1]];

	cvtColor(bin, bin, COLOR_GRAY2BGR);

	vector<Point> contour;
	for (int i = 0; i < hullcount; i++) {
		contour.push_back(points[hull[i]]);
	}

	area_convexHull = contourArea(contour);
	Point2f center;
	float radius;
	minEnclosingCircle(points, center, radius);
	area_circle = pi * radius*radius;
	ratio_section_to_convexHull = area_section / area_convexHull;
	ratio_section_to_circle = area_section / area_circle;
}

void reorder_tribe(vector<vector<Point2f>>& tribe_reorder, Mat last_layer, Mat& current_layer, vector<vector<Point2f>>& last_tribe) {
	Mat label;
	vector<vector<Point2f>> tribe;
	connectedComponents(current_layer, label, 8);
	connectedAreaBox(label, tribe);
	for (int i = 0; i < current_layer.rows; i++) {
		for (int j = 0; j < current_layer.cols; j++) {
			current_layer.at<uchar>(i, j) = 0;
		}
	}

	float layer_area_thresholdValue = 1.85;

	for (int i = 0; i < tribe.size(); i++) {
		for (int j = 0; j < tribe[i].size(); j++) {
			uchar data = last_layer.at<uchar>(tribe[i][j].y, tribe[i][j].x);
			if (data > 0) {
				float ratio_area_currentlLayer_lastLayer = float(tribe[i].size()) / float(last_tribe[data - 1].size());
				if (ratio_area_currentlLayer_lastLayer < layer_area_thresholdValue) {
					tribe_reorder[data - 1] = tribe[i];
					for (int k = 0; k < tribe[i].size(); k++) {
						current_layer.at<uchar>(tribe[i][k].y, tribe[i][k].x) = data;
					}
					break;
				}
			}
		}
	}
}


void main() {
	double resolution_XY = 0.07, resolution_Z = 0.7;
	int sliceNumber = 80;
	double culmVolume = 0;
	double culm_density = 0;
	double tillerNumber = 0;
	bool output_cluster = true;
	bool output_center = true;
	bool output_regression = true;
	string folder;
	cout << "Sample path: ";
	cin >> folder;

	string cluster_dstPath = folder + "\\cluster.txt";
	string center_path(folder + "\\center.txt");
	string regression_path(folder + "\\regression.txt");
	string surface(folder + "\\surface.ply");
	string detail_param(folder + "\\detail_param.txt");

	vector<string> imagePath;
	vector<string> originPath;


	ifstream fin(folder + "\\test.txt");
	string line_info, input_result;
	while (getline(fin, line_info)) {
		stringstream input(line_info);
		string origin, segmented;
		input >> origin;
		input >> segmented;
		imagePath.push_back(segmented);
		originPath.push_back(origin);
	}

	//convexhulll
	vector<float> vec_area_section;
	vector<float> vec_area_convexHull;
	vector<float> vec_area_circle;
	vector<float> vec_ratio_section_to_convexHull;
	vector<float> vec_ratio_section_to_circle;
	double area_section = 0;
	double area_convexHull = 0;
	double area_circle = 0;
	double ratio_section_to_convexHull = 0;
	double ratio_section_to_circle = 0;


	vector<vector<vector<Point2f>>> stem;
	string layer_s;
	double grayValue = 0;
	double pixelCount = 0;
	vector<double> tribeNum_at_each_layer;
	vector<Point3f> vec_outEdge;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

	//read the uppermost layer
	cout << "loading images: ";
	int progress = 1;
	printf("\rload images:   %d/%d", progress, sliceNumber);
	
	Mat img = imread(imagePath[imagePath.size() - 1], ImreadModes::IMREAD_GRAYSCALE);
	Mat origin = imread(originPath[originPath.size() - 1], ImreadModes::IMREAD_GRAYSCALE);
	culmVolume += countNonZero(img);

	Mat bin_;
	threshold(img, bin_, 128, 255, THRESH_BINARY_INV);
	deleteSmallArea(bin_, 50000);
	Mat outEdge;
	Canny(bin_, outEdge, 50, 1000);
	for (int i = 0; i < outEdge.rows; i++) {
		for (int j = 0; j < outEdge.cols; j++) {
			if (outEdge.at<uchar>(i, j)) {
				cloud->points.push_back(pcl::PointXYZ(j * resolution_XY, i * resolution_XY, 0));
			}
		}
	}
	

	_convexHull(img, area_section, area_convexHull, area_circle, ratio_section_to_convexHull, ratio_section_to_circle);
	vec_area_section.push_back(area_section);
	vec_area_convexHull.push_back(area_convexHull);
	vec_area_circle.push_back(area_circle);
	vec_ratio_section_to_convexHull.push_back(ratio_section_to_convexHull);
	vec_ratio_section_to_circle.push_back(ratio_section_to_circle);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) > 128) {
				grayValue += (255 - origin.at<uchar>(i, j));
				pixelCount++;
			}
		}
	}
	//tillerNumber

	Mat label;
	vector<vector<Point2f>> last_tribe;
	connectedComponents(img, label, 8);
	connectedAreaBox(label, last_tribe);
	stem.push_back(last_tribe);
	tribeNum_at_each_layer.push_back(last_tribe.size());

	Mat lastLayer(img.size(), CV_32F, Scalar::all(0));
	for (int i = 0; i < last_tribe.size(); i++) {
		for (int j = 0; j < last_tribe[i].size(); j++) {
			lastLayer.at<uchar>(last_tribe[i][j].y, last_tribe[i][j].x) = i + 1;
		}
	}

	vector<vector<Point2f>> tribe_reorder;

	//progress bar
	for (int layer = imagePath.size() - 2; layer >= 0; layer--) {
		progress++;
		if (progress / 10 > 0) {
			printf("\rload images:  %d/%d", progress, sliceNumber);
		}
		else {
			printf("\rload images:   %d/%d", progress, sliceNumber);
		}
		
		tribe_reorder.clear();
		tribe_reorder.resize(stem[0].size());
		Mat currentLayer = imread(imagePath[layer], ImreadModes::IMREAD_GRAYSCALE);
		origin = imread(originPath[layer], ImreadModes::IMREAD_GRAYSCALE);

		threshold(currentLayer, bin_, 128, 255, THRESH_BINARY_INV);
		deleteSmallArea(bin_, 50000);
		Canny(bin_, outEdge, 50, 1000);
		for (int i = 0; i < outEdge.rows; i++) {
			for (int j = 0; j < outEdge.cols; j++) {
				if (outEdge.at<uchar>(i, j)) {;
				cloud->points.push_back(pcl::PointXYZ(j * resolution_XY, i * resolution_XY, (sliceNumber - layer - 1) * resolution_Z));
				}
			}
		}

		//grayValue
		for (int i = 0; i < currentLayer.rows; i++) {
			for (int j = 0; j < currentLayer.cols; j++) {
				if (currentLayer.at<uchar>(i, j) > 128) {
					grayValue += (255 - origin.at<uchar>(i, j));
					pixelCount++;
				}
			}
		}

		_convexHull(currentLayer, area_section, area_convexHull, area_circle, ratio_section_to_convexHull, ratio_section_to_circle);
		vec_area_section.push_back(area_section);
		vec_area_convexHull.push_back(area_convexHull);
		vec_area_circle.push_back(area_circle);
		vec_ratio_section_to_convexHull.push_back(ratio_section_to_convexHull);
		vec_ratio_section_to_circle.push_back(ratio_section_to_circle);

		culmVolume += countNonZero(currentLayer);
		reorder_tribe(tribe_reorder, lastLayer, currentLayer, last_tribe);
		stem.push_back(tribe_reorder);
		lastLayer = currentLayer;
		last_tribe = tribe_reorder;
		tribeNum_at_each_layer.push_back(last_tribe.size());
	}
	cout << endl;

	double reconSurface(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float resolution_Z, string surface);
	cout << "Total_SA_culm = " << reconSurface(cloud, resolution_Z, surface) << endl;

	for (int i = 0; i < tribeNum_at_each_layer.size(); i++) {
		tillerNumber += tribeNum_at_each_layer[i];
	}
	tillerNumber /= tribeNum_at_each_layer.size();

	double ave_area_section = 0, ave_area_convexHull = 0, ave_area_circle = 0, ave_ratio_section_to_convexHull = 0, ave_ratio_section_to_circle = 0;
	for (int i = 0; i < vec_area_section.size(); i++) {
		ave_area_section += vec_area_section[i];
		ave_area_convexHull += vec_area_convexHull[i];
		ave_area_circle += vec_area_circle[i];
		ave_ratio_section_to_convexHull += vec_ratio_section_to_convexHull[i];
		ave_ratio_section_to_circle += vec_ratio_section_to_circle[i];
	}
	
	double global_area_culm = ave_area_section / (vec_area_section.size()*tillerNumber);

	ave_area_section /= vec_area_section.size();
	ave_area_convexHull /= vec_area_section.size();
	ave_area_circle /= vec_area_section.size();
	ave_ratio_section_to_convexHull /= vec_area_section.size();
	ave_ratio_section_to_circle /= vec_area_section.size();

	vector<vector<vector<Point2f>>> stem_reorder;
	stem_reorder.resize(stem[0].size());
	for (int i = 0; i < stem_reorder.size(); i++) {
		stem_reorder[i].resize(stem.size());
	}
	for (int layer = 0; layer < stem.size(); layer++) {
		for (int i = 0; i < stem[layer].size(); i++) {
			if (stem[layer][i].size()) {
				stem_reorder[i][layer] = stem[layer][i];
			}
		}
	}

	ofstream f_cluster(cluster_dstPath);
	if (output_cluster) {
		for (int i = 0; i < stem_reorder.size(); i++) {
			if (stem[i].size()) {
				int r = 255 * rand() / double(RAND_MAX);
				int g = 255 * rand() / double(RAND_MAX);
				int b = 255 * rand() / double(RAND_MAX);

				for (int layer = 0; layer < stem_reorder[i].size(); layer++) {
					for (int j = 0; j < stem_reorder[i][layer].size(); j++) {
						f_cluster << float(stem_reorder[i][layer][j].y)*resolution_XY << " " << float(stem_reorder[i][layer][j].x*resolution_XY) << " " << float(layer) * resolution_Z << " " << r << " " << g << " " << b << endl;
					}
				}
			}
		}
		cout << "done" << endl;
	}

	float axis_min = 10, axis_max = 100000;
	ofstream f_center(center_path);
	ofstream regression(regression_path);

	vector<vector<float>> param;
	vector<float> key;
	vector<Point2f> lineParam;
	for (int stem_i = 0; stem_i < stem_reorder.size(); stem_i++) {
		cout << "stem " << stem_i << ": ";
		vector<float> vecMajoraxisLength, vecMinoraxisLength, vecDiameter, vecWallThickness, vec_area, vec_area_out, vec_perimeter, vec_area_perimeter;
		Mat display(img.size(), CV_8UC3, Scalar(0, 0, 0));
		float major_layer0, minor_layer0, diameter0, wallThickness_layer0, area0, area_out0, perimeter0, area_perimeter0;
		getParamFromMat(img.size(), stem_reorder[stem_i][0], major_layer0, minor_layer0, wallThickness_layer0, area0, area_out0, perimeter0, area_perimeter0);
		diameter0 = (major_layer0 + minor_layer0) / 2;
		vecMajoraxisLength.push_back(major_layer0);
		vecMinoraxisLength.push_back(minor_layer0);
		vecDiameter.push_back(diameter0);
		vecWallThickness.push_back(wallThickness_layer0);
		vec_area.push_back(area0);
		vec_area_out.push_back(area_out0);
		vec_perimeter.push_back(perimeter0);
		vec_area_perimeter.push_back(area_perimeter0);

		float major_layer1, minor_layer1, diameter1, wallThickness_layer1, area1, area_out1, perimeter1, area_perimeter1;
		for (int layer = 1; layer < stem_reorder[stem_i].size(); layer++) {
			if (stem_reorder[stem_i][layer].size() > 0) {
				getParamFromMat(Size(1803, 1803), stem_reorder[stem_i][layer], major_layer1, minor_layer1, wallThickness_layer1, area1, area_out1, perimeter1, area_perimeter1);
				diameter1 = (major_layer1 + minor_layer1) / 2;
				float ratio = major_layer1 / major_layer0;
				if (ratio > 0 && ratio < 1.75
					&& major_layer1 > 0 && major_layer1 < 10000
					&& minor_layer1 > 0 && minor_layer1 < 10000
					&& diameter1 > 0 && diameter1 < 10000
					&& wallThickness_layer1 > 0 && wallThickness_layer1 < 10000
					&& area1 > 0 && area1 < 10000
					&& area_out1 > 0 && area_out1 < 10000
					&& perimeter1 > 0 && perimeter1 < 10000
					&& area_perimeter1 > 0 && area_perimeter1 < 10000) {

					vecMajoraxisLength.push_back(major_layer1);
					vecMinoraxisLength.push_back(minor_layer1);
					vecDiameter.push_back(diameter1);
					vecWallThickness.push_back(wallThickness_layer1);
					vec_area.push_back(area1);
					vec_area_out.push_back(area_out1);
					vec_perimeter.push_back(perimeter1);
					vec_area_perimeter.push_back(area_perimeter1);
				}
				else {
					for (int k = layer; k < stem_reorder[stem_i].size(); k++) {
						stem_reorder[stem_i][k].clear();
					}
					break;
				}
			}
		}

		if (vecMajoraxisLength.size() > 40) {
			float majoraxisLength = 0, minoraxisLength = 0, diameter = 0, wallThickness = 0, area = 0, area_out = 0, perimeter = 0, area_perimeter = 0;
			for (int i = 0; i < vecMajoraxisLength.size(); i++) {
				majoraxisLength += vecMajoraxisLength[i];
				minoraxisLength += vecMinoraxisLength[i];
				diameter += vecDiameter[i];
				wallThickness += vecWallThickness[i];
				area += vec_area[i];
				area_out += vec_area_out[i];
				perimeter += vec_perimeter[i];
				area_perimeter += vec_area_perimeter[i];
			}
			majoraxisLength /= vecMajoraxisLength.size();
			minoraxisLength /= vecMinoraxisLength.size();
			diameter /= vecDiameter.size();
			wallThickness /= vecWallThickness.size();
			area /= vec_area.size();
			area_out /= vec_area_out.size();
			perimeter /= vec_perimeter.size();
			area_perimeter /= vec_area_perimeter.size();
			cout << "number of counted layer: " << vecMajoraxisLength.size() << endl;

			//space line regression
			int r = 255 * rand() / double(RAND_MAX);
			int g = 255 * rand() / double(RAND_MAX);
			int b = 255 * rand() / double(RAND_MAX);
			vector<Point2f> center;
			center = extractCenterOfGravity(stem_reorder[stem_i]);
			double x0, y0, m, n;
			vector<Point3f> center_i;
			for (int j = 0; j < stem_reorder[stem_i].size(); j++) {
				if (stem_reorder[stem_i][j].size()) {
					f_center << center[j].y*resolution_XY << " " << center[j].x*resolution_XY << " " << j * resolution_Z << " " << r << " " << g << " " << b << endl;
					center_i.push_back(Point3f(center[j].y*resolution_XY, center[j].x*resolution_XY, j*resolution_Z));
				}
			}
			lineRegression(center_i, x0, y0, m, n);
			lineParam.push_back(Point2f(m, n));
			/*for (float z = 0; z < 100 * resolution_Z; z += 0.07) {
				regression << y0 + n * z << " " << x0 + m * z << " " << z << " " << r << " " << g << " " << b << endl;
			}*/
			for (float z = 0; z < sliceNumber * resolution_Z; z += resolution_Z) {
				regression << x0 + m * z << " " << y0 + n * z << " " << z << " " << r << " " << g << " " << b << endl;
			}
			float angle = angleBetweenLineAndPlanarXY(m, n);
			//float correction_coefficient = sin(angle*pi / 180);

			key.push_back(majoraxisLength);
			vector<float> param_temp;
			param_temp.push_back(minoraxisLength);//0
			param_temp.push_back(diameter);//1
			param_temp.push_back(wallThickness);//2
			param_temp.push_back(angle);//3
			param_temp.push_back(area);//4
			param_temp.push_back(area_out);//5
			param_temp.push_back(perimeter);//6
			param_temp.push_back(area_perimeter);//7
			param.push_back(param_temp);
			/*cout << "majoraxisLength_" << stem_i << " = " << majoraxisLength << endl;
			cout << "minoraxisLength_" << stem_i << " = " << minoraxisLength << endl;
			cout << "diameter_" << stem_i << " = " << diameter << endl;
			cout << "wallThickness_" << stem_i << " = " << wallThickness << endl;
			cout << "angle_" << stem_i << " = " << angle << endl;
			cout << "area_" << stem_i << " = " << area << endl;
			cout << "area_out_" << stem_i << " = " << area_out << endl;
			cout << "perimeter_" << stem_i << " = " << perimeter << endl;
			cout << "area_perimeter_" << stem_i << " = " << area_perimeter << endl << endl;
*/
		}
		else {
			cout << "ignored" << endl;
		}

	}

	cout << param.size() << " stems were counted" << endl;
	ofstream f_general_detail_param(detail_param);
	f_general_detail_param << "stemNum majorAxis minorAxis diameter wallThickness tillerAngle areaCulm areaCulmFilled perimeter ratio_area_perimeter" << endl;
	for (int i = 0; i < key.size(); i++) {
		f_general_detail_param<<"stem "<< i << " " << key[i] << " " << param[i][0] << " " << param[i][1] << " " << param[i][2] << " " << param[i][3] << " " << param[i][4] << " " << param[i][5] << " " << param[i][6] << " " << param[i][7] << endl;
	}
	
	vector<float> insert;
	float max, min, ave, V, SD;

	//area_culm
	for (int i = 0; i < param.size(); i++) {
		insert.push_back(param[i][4]);
	}
	max_min_V_SD(insert, max, min, ave, V, SD);
	cout << "Max_area_culm = " << max << endl;
	cout << "Mean_area_culm = " << ave << endl;
	cout << "SD_area_culm = " << SD << endl;
	insert.clear();

	//area_culm_to_perimeter
	for (int i = 0; i < param.size(); i++) {
		insert.push_back(param[i][7]);
	}
	max_min_V_SD(insert, max, min, ave, V, SD);
	cout << "Max_APR_culm = " << max << endl;
	cout << "Mean_APR_culm = " << ave << endl;
	cout << "SD_APR_culm = " << SD << endl;
	insert.clear();

	//tiller angle
	for (int i = 0; i < param.size(); i++) {
		insert.push_back(param[i][3]);
	}
	max_min_V_SD(insert, max, min, ave, V, SD);
	cout << "MENTA = " << ave << endl;
	cout << "MAXTA = " << max << endl;
	cout << "SDTA = " << SD << endl;
	insert.clear();

	vector<float> k1;
	sort(key, k1);
	vector<float> average_output(9, 0);
	if (k1.size() >= 3) {
		int count = 0;
		for (int i = k1.size() - 1; count < 3; i--) {
			for (int j = 0; j < key.size(); j++) {
				if (key[j] == k1[i]) {
					average_output[0] += k1[i];			//majoraxis length
					average_output[1] += param[j][0];	//minoraxis length
					average_output[2] += param[j][1];	//diameter
					average_output[3] += param[j][2];	//wall thickness
					average_output[4] += param[j][3];	//angle
					average_output[5] += param[j][4];	//culm section area
					average_output[6] += param[j][5];	//filled culm section area
					average_output[7] += param[j][6];	//perimeter
					average_output[8] += param[j][7];	//ratio of culm section area to perimeter

					//diameter
					insert.push_back(param[j][1]);
					count++;
					break;
				}
			}
		}
		//cout << "count: " << count << endl;
		//num area_section area_convexHull area_circle ratio_section_to_convexHull ratio_section_to_circle
		cout << "Total_volume_culm (culm volume): " << culmVolume << endl;
		cout << "global_area_culm = " << global_area_culm << endl;
		cout << "Total _area_culm) ave_area_section = " << ave_area_section << endl;
		cout << "CHA_culm (ave_area_convexHull) = " << ave_area_convexHull << endl;
		cout << "ave_area_circle = " << ave_area_circle << endl;
		cout << "CHR_culm (ave_ratio_section_to_convexHull) = " << ave_ratio_section_to_convexHull << endl;
		cout << "CCR_culm (ave_ratio_section_to_circle) = " << ave_ratio_section_to_circle << endl;
		cout << "Culm_density_total (grayValue) = " << grayValue << endl;
		cout << "Culm density_mean (culm density) = " << grayValue / pixelCount << endl;
		cout << "TN (tilletNumber) = " << tillerNumber << endl;
		//top3 diameter
		max_min_V_SD(insert, max, min, ave, V, SD);
		cout << "Mean_diameter_culm = " << ave << endl;
		cout << "Max_diameter_culm = " << max << endl;
		cout << "SD_diameter_culm = " << SD << endl;
		for (int i = 0; i < average_output.size(); i++) {
			average_output[i] /= 3;
		}
		cout << "Major_axis_culm = " << average_output[0] << endl;
		cout << "Minor_axis_culm = " << average_output[1] << endl;
		cout << "Wall_thickness_culm = " << average_output[3] << endl;
	}
	cin.get();
}
