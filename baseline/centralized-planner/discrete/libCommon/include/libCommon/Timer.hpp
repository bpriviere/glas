#pragma once

#include <chrono>
#include <iostream>

class Timer
{
public:
	Timer()
		: start_(std::chrono::high_resolution_clock::now())
		, end_(std::chrono::high_resolution_clock::now())
		, hasEnd_(false)
	{
	}

	void reset() {
		start_ = std::chrono::high_resolution_clock::now();
	}

	void stop() {
		end_ = std::chrono::high_resolution_clock::now();
		hasEnd_ = true;
	}

	double elapsedSeconds() {
		if (!hasEnd_) {
			stop();
		}
		auto timeSpan = std::chrono::duration_cast<std::chrono::duration<double>>(end_ - start_);
		return timeSpan.count();
	}

private:
	std::chrono::high_resolution_clock::time_point start_;
	std::chrono::high_resolution_clock::time_point end_;
	bool hasEnd_;
};

class ScopedTimer
	: public Timer
{
public:
	ScopedTimer(
		const std::string& desc)
		: m_desc(desc)
	{
	}

	~ScopedTimer()
	{
		stop();
		std::cout << m_desc << " Elapsed: " << elapsedSeconds() << " s" << std::endl;
	}

private:
	std::string m_desc;
};
