#include <iostream>
#include <arrayfire.h>
#include <memory>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>
#include <sstream>

const int N = 2000000;
const int GRID_DIM = 1024;
const float CELL_SIZE = 1.0;
const float G = 1.0;
const float DT = 0.02;

struct Vec2
{
	float x, y;
	Vec2(float x = 0, float y = 0) :
		x(x), y(y) { }
};

int wrapIndex(int i, int i_max) 
{
	return ((i % i_max) + i_max) % i_max;
}


af::array makeGrid(const Vec2* p, const float* m)
{
	float* grid_h = new float[GRID_DIM * GRID_DIM];
	memset(grid_h, 0, sizeof(float) * GRID_DIM * GRID_DIM);
	
	for(int i = 0; i < N; i++)
	{
		int x = wrapIndex(roundf(p[i].x / CELL_SIZE), GRID_DIM);
		int y = wrapIndex(roundf(p[i].y / CELL_SIZE), GRID_DIM);
		grid_h[y * GRID_DIM + x] += m[i];
	}
	
	af::array grid(GRID_DIM, GRID_DIM, grid_h);
	delete[] grid_h;
	return grid;
}

float randf()
{
	return (float)rand() / RAND_MAX * (rand() % 2 ? 1.0f : -1.0f);
}


int main()
{
	srand(time(0));
	af::setBackend(AF_BACKEND_CUDA);
	af::info();
	Vec2* p = new Vec2[N];
	Vec2* v = new Vec2[N];
	float* m = new float[N];
	
	std::fill(v, v + N, 0.0f);
	std::fill(m, m + N, 100.0f);
	
	for(int i = 0; i < N; i++)
		p[i] = Vec2(randf() * GRID_DIM, randf() * GRID_DIM);
	
//	for(int i = 0; i < N; i++)
//		v[i] = Vec2(50 * randf(), 50 * randf());
	
	for(int i = 0; i < N; i++)
		m[i] = (fabs(randf()) + 1) * 10;
	
	af::array rng = af::range(GRID_DIM);
	af::array k2 = 2.0 * af::Pi * af::select(rng > GRID_DIM / 2, (rng - GRID_DIM) , rng) / (GRID_DIM * CELL_SIZE);
	//af::array k2 = 2.0 * af::Pi * rng / (GRID_DIM * CELL_SIZE);
	k2 = af::tile((k2 * k2).T(), GRID_DIM) + af::tile(k2 * k2, 1, GRID_DIM);
	k2(0, 0) = 1.0;
	
	af::Window wnd(GRID_DIM, GRID_DIM);
	
	//load the grid with bodies
	af::array grid = makeGrid(p, m);
	int count = 0;
	while(!wnd.close())
	{
		af::timer t = af::timer::start();
		
		//fft the density grid
		grid = af::fft2(grid);
		
		//apply the formula from the paper
		grid *= -G / (af::Pi * k2);
		//invert the fft to get the potentials
		grid = af::ifft2(grid);
		grid = af::abs(grid);
		
		//compute the gradients in the potential
		af::array dx, dy;
		af::grad(dx, dy, grid);
		
		float* dx_h = dx.host<float>();
		float* dy_h = dy.host<float>();
		
		//use the gradients to time step each of the bodies
		#pragma omp parallel for simd
		for(int i = 0; i < N; i++)
		{
			int xi = wrapIndex(roundf(p[i].x / CELL_SIZE), GRID_DIM);
			int yi = wrapIndex(roundf(p[i].y / CELL_SIZE), GRID_DIM);
			
			v[i].x += dx_h[yi * GRID_DIM + xi] * DT / m[i];
			v[i].y += dy_h[yi * GRID_DIM + xi] * DT / m[i];
			p[i].x += v[i].x * DT;
			p[i].y += v[i].y * DT;
		}
		af::freeHost(dx_h);
		af::freeHost(dy_h);
		
		//display the potential
		
		
		//remake the new grid
		af::array img = af::abs(grid) / af::max<float>(af::abs(grid));
		wnd.image(img);
		grid = makeGrid(p, m);
		
		std::cout << "dt = " << af::timer::stop(t) << std::endl;
	}
	
	delete[] p;
	delete[] v;
	delete[] m;
	return 0;
}
