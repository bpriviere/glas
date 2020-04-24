#pragma once

template <typename T>
std::vector<T> asVector(
    const boost::property_tree::ptree& pt,
    const boost::property_tree::ptree::key_type& key)
{
    std::vector<T> r;
    for (auto& item : pt.get_child(key)) {
        r.push_back(item.second.get_value<T>());
    }
    return r;
}