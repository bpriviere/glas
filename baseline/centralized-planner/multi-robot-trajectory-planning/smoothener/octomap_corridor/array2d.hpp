#ifndef __ARRAY2D_HPP__
#define __ARRAY2D_HPP__

//
// StaticArray2D has fixed size at compile-time,
// DynamicArray2D has fixed size at construction.
//
// Memory is stored in a contiguous block.
// begin() and end() "iterators" allow std::algorithm compatibility.
//
// Operator() is used for 2D indexing.
// Oprerator[] is used for 1D indexing.
// Arrays are row-major: arr(3, 1) and arr(3, 2) are consecutive in memory.
//
//
// Copyright 2013 James Preiss.
// Public Domain - Attribution is appreciated.
//

#include <utility>

template <typename T, size_t Rows, size_t Columns>
class StaticArray2D
{
public:
	// default-construct all data
	void initialize()
	{
		for (size_t i = 0; i < Rows * Columns; ++i)
		{
			data_[i] = T();
		}
	}

	size_t rows() const
	{
		return Rows;
	}

	size_t columns() const
	{
		return Columns;
	}

	size_t to1D(size_t row, size_t column) const
	{
		return row * Columns + column;
	}

	std::pair<size_t, size_t> to2D(size_t index) const
	{
		return std::make_pair(index / Columns, index % Columns);
	}

	T const &operator[](size_t index) const
	{
		return data_[index];
	}

	T &operator[](size_t index)
	{
		return data_[index];
	}

	T const &operator()(size_t row, size_t column) const
	{
		return data_[to1D(row, column)];
	}

	T &operator()(size_t row, size_t column)
	{
		return data_[to1D(row, column)];
	}

	T const *data() const
	{
		return data_;
	}

	T *data()
	{
		return data_;
	}

	T const *begin() const
	{
		return data_;
	}

	T *begin()
	{
		return data_;
	}

	T const *end() const
	{
		return data_ + Rows * Columns;
	}

	T *end()
	{
		return data_ + Rows * Columns;
	}

private:
	T data_[Rows * Columns];
};


template <typename T>
class Array2D
{
public:
	Array2D(size_t rows, size_t columns) :
		rows_(rows),
		columns_(columns),
		data_(new T[rows * columns])
	{
	}

	Array2D(Array2D const &) = delete;

	Array2D(Array2D &&a) :
		rows_(a.rows_),
		columns_(a.columns_),
		data_(a.data_)
	{
		a.data_ = nullptr;
		a.rows_ = 0;
		a.columns_ = 0;
	}

	~Array2D()
	{
		delete[] data_;
	}

	size_t rows() const
	{
		return rows_;
	}

	size_t columns() const
	{
		return columns_;
	}

	size_t to1D(size_t row, size_t column) const
	{
		return row * columns_ + column;
	}

	std::pair<size_t, size_t> to2D(size_t index) const
	{
		return std::make_pair(index / columns_, index % columns_);
	}

	T const &operator[](size_t index) const
	{
		return data_[index];
	}

	T &operator[](size_t index)
	{
		return data_[index];
	}

	T const &operator()(size_t row, size_t column) const
	{
		return data_[to1D(row, column)];
	}

	T &operator()(size_t row, size_t column)
	{
		return data_[to1D(row, column)];
	}

	T const *data() const
	{
		return data_;
	}

	T *data()
	{
		return data_;
	}

	T const *begin() const
	{
		return data_;
	}

	T *begin()
	{
		return data_;
	}

	T const *end() const
	{
		return data_ + rows_ * columns_;
	}

	T *end()
	{
		return data_ + rows_ * columns_;
	}

private:
	size_t rows_;
	size_t columns_;
	T *data_;
};
#endif // __ARRAY2D_HPP__
