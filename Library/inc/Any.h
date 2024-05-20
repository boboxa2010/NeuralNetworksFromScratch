#pragma once

#include <memory>
#include <type_traits>

namespace nn {
namespace iternal {
template <template <typename> typename IAny, template <typename> typename Impl>
class Any {
    class Concept;

public:
    constexpr Any() noexcept = default;

    Any(const Any &rhs) : model_((rhs.HasValue() ? rhs->MakeCopy() : nullptr)) {
    }

    Any(Any &&rhs) noexcept = default;

    Any &operator=(const Any &rhs) {
        Any(rhs).Swap(*this);
        return *this;
    }

    Any &operator=(Any &&rhs) noexcept = default;

    template <typename T>
    Any(T &&value) : model_(std::make_unique<Impl<Keeper<T>>>(std::forward<T>(value))) {
    }

    template <class T>
    Any &operator=(T &&value) noexcept {
        Any(std::forward<T>(value)).Swap(*this);
        return *this;
    }

    bool HasValue() const {
        return model_ != nullptr;
    }

    void Clear() {
        model_.reset();
    }

    void Swap(Any &rhs) {
        std::swap(model_, rhs.model_);
    }

    Concept *operator->() {
        return model_.get();
    }

    const Concept *operator->() const {
        return model_.get();
    }

private:
    class IEmpty {
    protected:
        virtual ~IEmpty() = default;
    };

    class Concept : public IAny<IEmpty> {
        virtual std::unique_ptr<Concept> MakeCopy() const = 0;
        friend class Any;
    };

    template <typename T>
    class Keeper : public Concept {
    public:
        Keeper() = default;

        template <typename V>
        Keeper(V &&value) noexcept : data_(std::forward<V>(value)) {
        }

    protected:
        T &Get() {
            return data_;
        }

        const T &Get() const {
            return data_;
        }

    private:
        std::unique_ptr<Concept> MakeCopy() const override {
            return std::make_unique<Impl<Keeper<T>>>(data_);
        }

        T data_;
    };

    std::unique_ptr<Concept> model_;
};
}  // namespace iternal
}  // namespace nn
