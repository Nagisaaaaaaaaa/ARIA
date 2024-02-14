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
  auto expectNChildren = [] (Object& o, size_t num) {
    size_t i = 0;
    for (auto & child : o) {
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
    EXPECT_TRUE(o.IsRoot());
    EXPECT_TRUE(&oPare != &o);
    EXPECT_EQ(&oRoot, &o);

    Object &o1 = Object::Create();
    o1.parent() = &o;
    const Object &o1Pare = *o1.parent();
    const Object &o1Root = *o1.root();
    EXPECT_FALSE(o1.IsRoot());
    EXPECT_EQ(&o1Pare, &o);
    EXPECT_EQ(&o1Root, &o);

    Object &o2 = Object::Create();
    o2.parent() = &o1;
    const Object &o2Pare = *o2.parent();
    const Object &o2Root = *o2.root();
    EXPECT_FALSE(o2.IsRoot());
    EXPECT_EQ(&o2Pare, &o1);
    EXPECT_EQ(&o2Root, &o);
  }

  {
    Object &o = Object::Create();
    const Object &oPare = *o.parent();
    const Object &oRoot = *o.root();
    Transform &t = o.transform();
    Transform &tPare = *t.parent();
    Transform &tRoot = *t.root();
    EXPECT_TRUE(t.IsRoot());
    EXPECT_TRUE(&tPare != &t);
    EXPECT_EQ(&tRoot, &t);

    Object &o1 = Object::Create();
    o1.parent() = &o;
    const Object &o1Pare = *o1.parent();
    const Object &o1Root = *o1.root();
    Transform &t1 = o1.transform();
    Transform &t1Pare = *t1.parent();
    Transform &t1Root = *t1.root();
    EXPECT_FALSE(t1.IsRoot());
    EXPECT_EQ(&t1Pare, &t);
    EXPECT_EQ(&t1Root, &t);

    Object &o2 = Object::Create();
    o2.parent() = &o1;
    const Object &o2Pare = *o2.parent();
    const Object &o2Root = *o2.root();
    Transform &t2 = o2.transform();
    Transform &t2Pare = *t2.parent();
    Transform &t2Root = *t2.root();
    EXPECT_FALSE(t2.IsRoot());
    EXPECT_EQ(&t2Pare, &t1);
    EXPECT_EQ(&t2Root, &t);
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

    EXPECT_EQ(o1.parent(), &o0);
    EXPECT_EQ(o2.parent(), &o1);
    EXPECT_EQ(o3.parent(), &o0);
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

    EXPECT_EQ(o1.parent(), &o0);
    EXPECT_EQ(o2.parent(), &o0);
    EXPECT_EQ(o3.parent(), &o2);
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

    EXPECT_EQ(o1.parent(), &o0);
    EXPECT_EQ(o2.parent(), &o1);
    EXPECT_EQ(o3.parent(), &o0);

    EXPECT_EQ(o1.transform().parent(), &o0.transform());
    EXPECT_EQ(o2.transform().parent(), &o1.transform());
    EXPECT_EQ(o3.transform().parent(), &o0.transform());
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

    EXPECT_EQ(o1.parent(), &o0);
    EXPECT_EQ(o2.parent(), &o0);
    EXPECT_EQ(o3.parent(), &o2);

    EXPECT_EQ(o1.transform().parent(), &o0.transform());
    EXPECT_EQ(o2.transform().parent(), &o0.transform());
    EXPECT_EQ(o3.transform().parent(), &o2.transform());
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

    EXPECT_EQ(o1.parent(), &o0);
    EXPECT_EQ(o2.parent(), &o0);
    EXPECT_EQ(o3.parent(), &o2);
  }

  {
    Object &o0 = Object::Create();
    Object &o1 = Object::Create();
    Object &o2 = Object::Create();
    Object &o3 = Object::Create();

    o1.parent() = &o0;
    o3.parent() = &o2;

    o3.root() = &o0;

    EXPECT_EQ(o1.parent(), &o0);
    EXPECT_EQ(o2.parent(), &o0);
    EXPECT_EQ(o3.parent(), &o2);
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

    EXPECT_EQ(o1.parent(), &o0);
    EXPECT_EQ(o2.parent(), &o0);
    EXPECT_EQ(o3.parent(), &o2);

    EXPECT_EQ(o1.transform().parent(), &o0.transform());
    EXPECT_EQ(o2.transform().parent(), &o0.transform());
    EXPECT_EQ(o3.transform().parent(), &o2.transform());
  }

  {
    Object &o0 = Object::Create();
    Object &o1 = Object::Create();
    Object &o2 = Object::Create();
    Object &o3 = Object::Create();

    o1.transform().parent() = &o0.transform();
    o3.transform().parent() = &o2.transform();

    o3.transform().root() = &o0.transform();

    EXPECT_EQ(o1.parent(), &o0);
    EXPECT_EQ(o2.parent(), &o0);
    EXPECT_EQ(o3.parent(), &o2);

    EXPECT_EQ(o1.transform().parent(), &o0.transform());
    EXPECT_EQ(o2.transform().parent(), &o0.transform());
    EXPECT_EQ(o3.transform().parent(), &o2.transform());
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
}

TEST(Object, Ranges) {
  // Range.
  {
    Object &o0 = Object::Create();
    o0.name() = "o0";
    TestComp &c00 = o0.AddComponent<TestComp>();
    Object &o1 = Object::Create();
    o1.name() = "o1";
    TestComp &c10 = o1.AddComponent<TestComp>();
    TestComp &c11 = o1.AddComponent<TestComp>();
    Object &o2 = Object::Create();
    o2.name() = "o2";
    TestComp &c20 = o2.AddComponent<TestComp>();
    TestComp &c21 = o2.AddComponent<TestComp>();
    Object &o3 = Object::Create();
    o3.name() = "o3";
    TestComp &c30 = o3.AddComponent<TestComp>();
    Object &o4 = Object::Create();
    o4.name() = "o4";
    TestComp &c40 = o4.AddComponent<TestComp>();
    Object &o5 = Object::Create();
    o5.name() = "o5";
    TestComp &c50 = o5.AddComponent<TestComp>();
    TestComp &c51 = o5.AddComponent<TestComp>();
    Object &o6 = Object::Create();
    o6.name() = "o6";
    TestComp &c60 = o6.AddComponent<TestComp>();
    TestComp &c61 = o6.AddComponent<TestComp>();

    o6.parent() = &o3;

    o5.parent() = &o2;

    o3.parent() = &o1;
    o4.parent() = &o1;

    o1.parent() = &o0;
    o2.parent() = &o0;

    std::stack<std::string> nameStack;

#if 0
    for (auto &obj : Object::rangeJoined()) {
      nameStack.push(obj.name());
    }

    EXPECT_EQ(nameStack.top(), "o5");
    nameStack.pop();
    EXPECT_EQ(nameStack.top(), "o2");
    nameStack.pop();
    EXPECT_EQ(nameStack.top(), "o4");
    nameStack.pop();
    EXPECT_EQ(nameStack.top(), "o6");
    nameStack.pop();
    EXPECT_EQ(nameStack.top(), "o3");
    nameStack.pop();
    EXPECT_EQ(nameStack.top(), "o1");
    nameStack.pop();
    EXPECT_EQ(nameStack.top(), "o0");
    nameStack.pop();
#endif
  }
}

} // namespace ARIA
