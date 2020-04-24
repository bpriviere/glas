#pragma once

#include <chrono>
#include <vector>
#include <iostream>

class StopWatch
{
public:
	StopWatch()
		: startTime(std::chrono::high_resolution_clock::now())
		, stopTime(std::chrono::high_resolution_clock::now())
		, stopped(false)
	{
	}

	void start(){
		stopped = false;
		startTime = std::chrono::high_resolution_clock::now();
	}

	double stop() {
		stopped = true;
		stopTime = std::chrono::high_resolution_clock::now();
		return seconds();
	}

	//resets start/stop and laps. timer not put into stop mode
	void reset() {
		stopped = false;
		startTime = std::chrono::high_resolution_clock::now();
		stopTime = std::chrono::high_resolution_clock::now();
		laps.clear();
	}

	//adds seconds to lap
	double lap(){
		laps.push_back(seconds() - laps.back());
		return laps.back();
	}

	//current stop watch time; if stopped returns stop-start, else returns now-start
	double seconds() {
		if (stopped){
			auto timeSpan = std::chrono::duration_cast<std::chrono::duration<double>>(stopTime - startTime);
			return timeSpan.count();
		}

		auto timeSpan = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - startTime);
		return timeSpan.count();
	}

	//Holds lap times
	std::vector<double> laps;

private:
	std::chrono::high_resolution_clock::time_point startTime;
	std::chrono::high_resolution_clock::time_point stopTime;
	bool stopped;
};
