#include "ARIA/Scene/Object.h"
#include "ARIA/Scene/Components/Transform.h"

#include <gtest/gtest.h>

#include <stack>

namespace ARIA {

namespace {

class TestComp final : public Component {
private:
  friend Object;

  using Base = Component;
  using Base::Base;

public:
  TestComp(const TestComp &) = delete;
  TestComp(TestComp &&) noexcept = delete;
  TestComp &operator=(const TestComp &) = delete;
  TestComp &operator=(TestComp &&) noexcept = delete;
  ~TestComp() final = default;
};

} // namespace

TEST(Object, Base) {
  auto expectSize = [](size_t num) {
    EXPECT_EQ(Object::size(), num);
    size_t i = 0;
    for (auto &o : Object::CRange())
      ++i;
    EXPECT_EQ(i, num);
  };
  auto expectNChildren = [](Object &o, size_t num) {
    size_t i = 0;
    for (auto &child : o) {
      std::string s = child.name(); // Indirect iterator.
      ++i;
    }
    EXPECT_EQ(i, num);
  };

  // Simple create, destroy immediate, registry, and iterator.
  //! Destroy all unrelated objects.
  while (Object::size() > 0) {
    DestroyImmediate(*Object::Begin());
  }
  EXPECT_EQ(Object::size(), 0);

  {
    EXPECT_EQ(Object::size(), 0);
    {
      int i = 0;
      for (auto it = Object::Begin(); it != Object::End(); ++it)
        ++i;
      EXPECT_EQ(i, 0);
    }
    Object &o = Object::Create();
    EXPECT_EQ(Object::size(), 1);
    {
      int i = 0;
      for (auto &o : Object::Range())
        ++i;
      EXPECT_EQ(i, 1);
    }
    {
      size_t i = 0;
      for (auto it = o.begin(); it != o.end(); ++it) {
        std::string s = it->name();
        ++i;
      }
      EXPECT_EQ(i, 0);
    }
    { // Try destroying halo root.
      bool failed = false;
      try {
        DestroyImmediate(*o.parent());
      } catch (std::exception &e) { failed = true; }
      EXPECT_TRUE(failed);
    }
    DestroyImmediate(o);
    EXPECT_EQ(Object::size(), 0);
    {
      int i = 0;
      for (auto &o : Object::CRange())
        ++i;
      EXPECT_EQ(i, 0);
    }
  }

  // Complex create, destroy immediate, registry, and iterator.
  //! Destroy all unrelated objects.
  {
    expectSize(0);
    Object &o0 = Object::Create();
    expectSize(1);
    expectNChildren(o0, 0);
    Object &o1 = Object::Create();
    expectSize(2);
    expectNChildren(o0, 0);
    o1.parent() = &o0;
    expectSize(2);
    expectNChildren(o0, 1);
    DestroyImmediate(o0);
    expectSize(0);
  }

  {
    expectSize(0);
    Object &o0 = Object::Create();
    expectSize(1);
    expectNChildren(o0, 0);
    Object &o1 = Object::Create();
    expectSize(2);
    expectNChildren(o0, 0);
    o1.parent() = &o0;
    expectSize(2);
    expectNChildren(o0, 1);
    DestroyImmediate(o1);
    expectSize(1);
    expectNChildren(o0, 0);
    DestroyImmediate(o0);
    expectSize(0);
  }

  {
    expectSize(0);
    Object &o0 = Object::Create();
    expectSize(1);
    expectNChildren(o0, 0);
    Object &o1 = Object::Create();
    expectSize(2);
    expectNChildren(o0, 0);
    Object &o2 = Object::Create();
    expectSize(3);
    expectNChildren(o0, 0);
    o1.parent() = &o0;
    expectSize(3);
    expectNChildren(o0, 1);
    o2.parent() = &o1;
    expectSize(3);
    expectNChildren(o0, 1);
    DestroyImmediate(o0);
    expectSize(0);
  }

  {
    expectSize(0);
    Object &o0 = Object::Create();
    expectSize(1);
    expectNChildren(o0, 0);
    Object &o1 = Object::Create();
    expectSize(2);
    expectNChildren(o0, 0);
    Object &o2 = Object::Create();
    expectSize(3);
    expectNChildren(o0, 0);
    o1.parent() = &o0;
    expectSize(3);
    expectNChildren(o0, 1);
    o2.parent() = &o1;
    expectSize(3);
    expectNChildren(o0, 1);
    DestroyImmediate(o1);
    expectSize(1);
    expectNChildren(o0, 0);
    DestroyImmediate(o0);
    expectSize(0);
  }

  {
    expectSize(0);
    Object &o0 = Object::Create();
    expectSize(1);
    expectNChildren(o0, 0);
    Object &o1 = Object::Create();
    expectSize(2);
    expectNChildren(o0, 0);
    Object &o2 = Object::Create();
    expectSize(3);
    expectNChildren(o0, 0);
    o1.parent() = &o0;
    expectSize(3);
    expectNChildren(o0, 1);
    o2.parent() = &o1;
    expectSize(3);
    expectNChildren(o0, 1);
    DestroyImmediate(o2);
    expectSize(2);
    expectNChildren(o0, 1);
    DestroyImmediate(o0);
    expectSize(0);
  }

  {
    expectSize(0);
    Object &o0 = Object::Create();
    expectSize(1);
    expectNChildren(o0, 0);
    Object &o1 = Object::Create();
    expectSize(2);
    expectNChildren(o0, 0);
    Object &o2 = Object::Create();
    expectSize(3);
    expectNChildren(o0, 0);
    o1.parent() = &o0;
    expectSize(3);
    expectNChildren(o0, 1);
    o2.parent() = &o0;
    expectSize(3);
    expectNChildren(o0, 2);
    DestroyImmediate(o0);
    expectSize(0);
  }

  // Name.
  {
    Object &o = Object::Create();
    o.name() = "0";
    EXPECT_EQ(o.name(), "0");
  }
}

TEST(Object, ParentRootAndTransform) {
  // Basic parents and roots.
  {
    Object &o = Object::Create();
    const Object &oPare = *o.parent();
    const Object &oRoot = *o.root();
    EXPECT_TRUE(oPare != o);
    EXPECT_TRUE(oRoot == o);

    Object &o1 = Object::Create();
    o1.parent() = &o;
    const Object &o1Pare = *o1.parent();
    const Object &o1Root = *o1.root();
    EXPECT_TRUE(o1Pare == o);
    EXPECT_TRUE(o1Root == o);

    Object &o2 = Object::Create();
    o2.parent() = &o1;
    const Object &o2Pare = *o2.parent();
    const Object &o2Root = *o2.root();
    EXPECT_TRUE(o2Pare == o1);
    EXPECT_TRUE(o2Root == o);
  }

  {
    Object &o = Object::Create();
    const Object &oPare = *o.parent();
    const Object &oRoot = *o.root();
    Transform &t = o.transform();
    Transform &tPare = *t.parent();
    Transform &tRoot = *t.root();
    EXPECT_TRUE(tPare != t);
    EXPECT_TRUE(tRoot == t);

    Object &o1 = Object::Create();
    o1.parent() = &o;
    const Object &o1Pare = *o1.parent();
    const Object &o1Root = *o1.root();
    Transform &t1 = o1.transform();
    Transform &t1Pare = *t1.parent();
    Transform &t1Root = *t1.root();
    EXPECT_TRUE(t1Pare == t);
    EXPECT_TRUE(t1Root == t);

    Object &o2 = Object::Create();
    o2.parent() = &o1;
    const Object &o2Pare = *o2.parent();
    const Object &o2Root = *o2.root();
    Transform &t2 = o2.transform();
    Transform &t2Pare = *t2.parent();
    Transform &t2Root = *t2.root();
    EXPECT_TRUE(t2Pare == t1);
    EXPECT_TRUE(t2Root == t);
  }

  // Complex parents.
  {
    Object &o0 = Object::Create();
    Object &o1 = Object::Create();
    Object &o2 = Object::Create();
    Object &o3 = Object::Create();

    o1.parent() = &o0;
    o2.parent() = &o1;
    o3.parent() = &o2;

    o3.parent() = &o0;

    EXPECT_TRUE(*o1.parent() == o0);
    EXPECT_TRUE(*o2.parent() == o1);
    EXPECT_TRUE(*o3.parent() == o0);
  }

  {
    Object &o0 = Object::Create();
    Object &o1 = Object::Create();
    Object &o2 = Object::Create();
    Object &o3 = Object::Create();

    o1.parent() = &o0;
    o2.parent() = &o1;
    o3.parent() = &o2;

    o2.parent() = &o0;

    EXPECT_TRUE(*o1.parent() == o0);
    EXPECT_TRUE(*o2.parent() == o0);
    EXPECT_TRUE(*o3.parent() == o2);
  }

  {
    Object &o0 = Object::Create();
    Object &o1 = Object::Create();
    Object &o2 = Object::Create();
    Object &o3 = Object::Create();

    o1.parent() = &o0;
    o2.parent() = &o1;
    o3.parent() = &o2;

    bool failed = false;
    try {
      o2.parent() = &o3;
    } catch (std::exception &e) { failed = true; }

    EXPECT_TRUE(failed);
  }

  {
    Object &o0 = Object::Create();
    Object &o1 = Object::Create();
    Object &o2 = Object::Create();
    Object &o3 = Object::Create();

    o1.transform().parent() = &o0.transform();
    o2.transform().parent() = &o1.transform();
    o3.transform().parent() = &o2.transform();

    o3.transform().parent() = &o0.transform();

    EXPECT_TRUE(*o1.parent() == o0);
    EXPECT_TRUE(*o2.parent() == o1);
    EXPECT_TRUE(*o3.parent() == o0);

    EXPECT_TRUE(*o1.transform().parent() == o0.transform());
    EXPECT_TRUE(*o2.transform().parent() == o1.transform());
    EXPECT_TRUE(*o3.transform().parent() == o0.transform());
  }

  {
    Object &o0 = Object::Create();
    Object &o1 = Object::Create();
    Object &o2 = Object::Create();
    Object &o3 = Object::Create();

    o1.transform().parent() = &o0.transform();
    o2.transform().parent() = &o1.transform();
    o3.transform().parent() = &o2.transform();

    o2.transform().parent() = &o0.transform();

    EXPECT_TRUE(*o1.parent() == o0);
    EXPECT_TRUE(*o2.parent() == o0);
    EXPECT_TRUE(*o3.parent() == o2);

    EXPECT_TRUE(*o1.transform().parent() == o0.transform());
    EXPECT_TRUE(*o2.transform().parent() == o0.transform());
    EXPECT_TRUE(*o3.transform().parent() == o2.transform());
  }

  {
    Object &o0 = Object::Create();
    Object &o1 = Object::Create();
    Object &o2 = Object::Create();
    Object &o3 = Object::Create();

    o1.transform().parent() = &o0.transform();
    o2.transform().parent() = &o1.transform();
    o3.transform().parent() = &o2.transform();

    bool failed = false;
    try {
      o2.transform().parent() = &o3.transform();
    } catch (std::exception &e) { failed = true; }

    EXPECT_TRUE(failed);
  }

  // Complex roots.
  {
    Object &o0 = Object::Create();
    Object &o1 = Object::Create();
    Object &o2 = Object::Create();
    Object &o3 = Object::Create();

    o1.parent() = &o0;
    o3.parent() = &o2;

    o2.root() = &o0;

    EXPECT_TRUE(*o1.parent() == o0);
    EXPECT_TRUE(*o2.parent() == o0);
    EXPECT_TRUE(*o3.parent() == o2);
  }

  {
    Object &o0 = Object::Create();
    Object &o1 = Object::Create();
    Object &o2 = Object::Create();
    Object &o3 = Object::Create();

    o1.parent() = &o0;
    o3.parent() = &o2;

    o3.root() = &o0;

    EXPECT_TRUE(*o1.parent() == o0);
    EXPECT_TRUE(*o2.parent() == o0);
    EXPECT_TRUE(*o3.parent() == o2);
  }

  {
    Object &o0 = Object::Create();
    Object &o1 = Object::Create();
    Object &o2 = Object::Create();
    Object &o3 = Object::Create();

    o1.parent() = &o0;
    o2.parent() = &o1;
    o3.parent() = &o2;

    bool failed = false;
    try {
      o2.root() = &o3;
    } catch (std::exception &e) { failed = true; }

    EXPECT_TRUE(failed);
  }

  {
    Object &o0 = Object::Create();
    Object &o1 = Object::Create();
    Object &o2 = Object::Create();
    Object &o3 = Object::Create();

    o1.transform().parent() = &o0.transform();
    o3.transform().parent() = &o2.transform();

    o2.transform().root() = &o0.transform();

    EXPECT_TRUE(*o1.parent() == o0);
    EXPECT_TRUE(*o2.parent() == o0);
    EXPECT_TRUE(*o3.parent() == o2);

    EXPECT_TRUE(*o1.transform().parent() == o0.transform());
    EXPECT_TRUE(*o2.transform().parent() == o0.transform());
    EXPECT_TRUE(*o3.transform().parent() == o2.transform());
  }

  {
    Object &o0 = Object::Create();
    Object &o1 = Object::Create();
    Object &o2 = Object::Create();
    Object &o3 = Object::Create();

    o1.transform().parent() = &o0.transform();
    o3.transform().parent() = &o2.transform();

    o3.transform().root() = &o0.transform();

    EXPECT_TRUE(*o1.parent() == o0);
    EXPECT_TRUE(*o2.parent() == o0);
    EXPECT_TRUE(*o3.parent() == o2);

    EXPECT_TRUE(*o1.transform().parent() == o0.transform());
    EXPECT_TRUE(*o2.transform().parent() == o0.transform());
    EXPECT_TRUE(*o3.transform().parent() == o2.transform());
  }

  {
    Object &o0 = Object::Create();
    Object &o1 = Object::Create();
    Object &o2 = Object::Create();
    Object &o3 = Object::Create();

    o1.transform().parent() = &o0.transform();
    o2.transform().parent() = &o1.transform();
    o3.transform().parent() = &o2.transform();

    bool failed = false;
    try {
      o2.transform().root() = &o3.transform();
    } catch (std::exception &e) { failed = true; }

    EXPECT_TRUE(failed);
  }

  // Sub-properties of parent and root.
  {
    Object &o0 = Object::Create();
    Object &o1 = Object::Create();
    Object &o2 = Object::Create();
    Object &o3 = Object::Create();

    o1.parent() = &o0;
    o2.parent() = &o1;
    o3.parent() = &o2;

    // Level 2.
    EXPECT_TRUE(*o2.parent()->parent() == o0);
    EXPECT_TRUE(*o2.parent()->root() == o0);
    EXPECT_TRUE(o2.parent()->transform() == o1.transform());

    EXPECT_TRUE(*o2.root()->parent() == *o0.parent());
    EXPECT_TRUE(*o2.root()->root() == o0);
    EXPECT_TRUE(o2.root()->transform() == o0.transform());

    EXPECT_TRUE(*o3.parent()->parent() == o1);
    EXPECT_TRUE(*o3.parent()->root() == o0);
    EXPECT_TRUE(o3.parent()->transform() == o2.transform());

    EXPECT_TRUE(*o3.root()->parent() == *o0.parent());
    EXPECT_TRUE(*o3.root()->root() == o0);
    EXPECT_TRUE(o3.root()->transform() == o0.transform());

    // Level 3.
    EXPECT_TRUE(*o2.parent()->parent()->parent() == *o0.parent());
    EXPECT_TRUE(o2.parent()->parent()->transform() == o0.transform());

    EXPECT_TRUE(*o2.parent()->root()->parent() == *o0.parent());
    EXPECT_TRUE(*o2.parent()->root()->root() == o0);
    EXPECT_TRUE(o2.parent()->root()->transform() == o0.transform());

    //
    EXPECT_TRUE(o2.root()->parent()->transform() == o0.parent().transform());

    EXPECT_TRUE(*o2.root()->root()->parent() == *o0.parent());
    EXPECT_TRUE(*o2.root()->root()->root() == o0);
    EXPECT_TRUE(o2.root()->root()->transform() == o0.transform());

    //
    EXPECT_TRUE(*o3.parent()->parent()->parent() == o0);
    EXPECT_TRUE(o3.parent()->parent()->transform() == o1.transform());

    EXPECT_TRUE(*o3.parent()->root()->parent() == *o0.parent());
    EXPECT_TRUE(*o3.parent()->root()->root() == o0);
    EXPECT_TRUE(o3.parent()->root()->transform() == o0.transform());

    //
    EXPECT_TRUE(o3.root()->parent()->transform() == o0.parent().transform());

    EXPECT_TRUE(*o3.root()->root()->parent() == *o0.parent());
    EXPECT_TRUE(*o3.root()->root()->root() == o0);
    EXPECT_TRUE(o3.root()->root()->transform() == o0.transform());
  }

  // Is root and is child of.
  {
    Object &o0 = Object::Create();
    Object &o1 = Object::Create();
    Object &o2 = Object::Create();
    Object &o3 = Object::Create();

    o1.parent() = &o0;
    o2.parent() = &o1;
    o3.parent() = &o2;

    EXPECT_TRUE(o0.IsRoot());
    EXPECT_FALSE(o1.IsRoot());
    EXPECT_FALSE(o2.IsRoot());
    EXPECT_FALSE(o3.IsRoot());

    EXPECT_TRUE(o1.IsChildOf(o0));
    EXPECT_TRUE(o2.IsChildOf(o0));
    EXPECT_TRUE(o2.IsChildOf(o1));
    EXPECT_TRUE(o3.IsChildOf(o0));
    EXPECT_TRUE(o3.IsChildOf(o1));
    EXPECT_TRUE(o3.IsChildOf(o2));

    EXPECT_FALSE(o0.IsChildOf(o0));
    EXPECT_FALSE(o0.IsChildOf(o1));
    EXPECT_FALSE(o0.IsChildOf(o2));
    EXPECT_FALSE(o0.IsChildOf(o3));
    EXPECT_FALSE(o1.IsChildOf(o1));
    EXPECT_FALSE(o1.IsChildOf(o2));
    EXPECT_FALSE(o1.IsChildOf(o3));
    EXPECT_FALSE(o2.IsChildOf(o2));
    EXPECT_FALSE(o2.IsChildOf(o3));
    EXPECT_FALSE(o3.IsChildOf(o3));
  }
}

TEST(Object, AddDestroyAndGetComponents) {
  // Transform.
  {
    Object& o = Object::Create();

    // Get.
    Transform* t = o.GetComponent<Transform>();
    EXPECT_NE(t, nullptr);

    // Destroy.
    {
      bool failed = false;
      try {
        DestroyImmediate(o.transform());
      } catch (std::exception &e) { failed = true; }
      EXPECT_TRUE(failed);
    }
  }

  // Others.
  {
    Object& o = Object::Create();

    // Get before add.
    EXPECT_EQ(o.GetComponent<TestComp>(), nullptr);

    // Add.
    TestComp& test = o.AddComponent<TestComp>();

    // Get.
    EXPECT_EQ(o.GetComponent<TestComp>(), &test);

    // Get transform.
    Transform* t = o.GetComponent<Transform>();
    EXPECT_NE(t, nullptr);

    // Destroy transform.
    {
      bool failed = false;
      try {
        DestroyImmediate(o.transform());
      } catch (std::exception &e) { failed = true; }
      EXPECT_TRUE(failed);
    }

    // Destroy.
    DestroyImmediate(test);

    // Get after destruction.
    EXPECT_EQ(o.GetComponent<TestComp>(), nullptr);
  }
}

} // namespace ARIA
