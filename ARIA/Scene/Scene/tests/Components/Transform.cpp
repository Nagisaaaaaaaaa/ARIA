#include "ARIA/Scene/Components/Transform.h"
#include "ARIA/Scene/Object.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {} // namespace

TEST(Transform, ParentRootAndTransform) {
  // Basic parents and roots.
  {
    Object &o = Object::Create();
    Transform &t = o.transform();
    const Transform &tPare = *t.parent();
    const Transform &tRoot = *t.root();
    EXPECT_TRUE(tPare != t);
    EXPECT_TRUE(tRoot == t);

    Object &o1 = Object::Create();
    Transform &t1 = o1.transform();
    t1.parent() = &t;
    const Transform &t1Pare = *t1.parent();
    const Transform &t1Root = *t1.root();
    EXPECT_TRUE(t1Pare == t);
    EXPECT_TRUE(t1Root == t);

    Object &o2 = Object::Create();
    Transform &t2 = o2.transform();
    t2.parent() = &t1;
    const Transform &t2Pare = *t2.parent();
    const Transform &t2Root = *t2.root();
    EXPECT_TRUE(t2Pare == t1);
    EXPECT_TRUE(t2Root == t);
  }

  // Complex parents.
  {
    Object &o0 = Object::Create();
    Object &o1 = Object::Create();
    Object &o2 = Object::Create();
    Object &o3 = Object::Create();

    Transform &t0 = o0.transform();
    Transform &t1 = o1.transform();
    Transform &t2 = o2.transform();
    Transform &t3 = o3.transform();

    t1.parent() = &t0;
    t2.parent() = &t1;
    t3.parent() = &t2;

    t3.parent() = &t0;

    EXPECT_TRUE(*t1.parent() == t0);
    EXPECT_TRUE(*t2.parent() == t1);
    EXPECT_TRUE(*t3.parent() == t0);
  }

  {
    Object &o0 = Object::Create();
    Object &o1 = Object::Create();
    Object &o2 = Object::Create();
    Object &o3 = Object::Create();

    Transform &t0 = o0.transform();
    Transform &t1 = o1.transform();
    Transform &t2 = o2.transform();
    Transform &t3 = o3.transform();

    t1.parent() = &t0;
    t2.parent() = &t1;
    t3.parent() = &t2;

    t2.parent() = &t0;

    EXPECT_TRUE(*t1.parent() == t0);
    EXPECT_TRUE(*t2.parent() == t0);
    EXPECT_TRUE(*t3.parent() == t2);
  }

  {
    Object &o0 = Object::Create();
    Object &o1 = Object::Create();
    Object &o2 = Object::Create();
    Object &o3 = Object::Create();

    Transform &t0 = o0.transform();
    Transform &t1 = o1.transform();
    Transform &t2 = o2.transform();
    Transform &t3 = o3.transform();

    t1.parent() = &t0;
    t2.parent() = &t1;
    t3.parent() = &t2;

    bool failed = false;
    try {
      t2.parent() = &t3;
    } catch (std::exception &e) { failed = true; }

    EXPECT_TRUE(failed);
  }

  {
    Object &o0 = Object::Create();
    Object &o1 = Object::Create();
    Object &o2 = Object::Create();
    Object &o3 = Object::Create();

    Transform &t0 = o0.transform();
    Transform &t1 = o1.transform();
    Transform &t2 = o2.transform();
    Transform &t3 = o3.transform();

    t1.transform().parent() = &t0.transform();
    t2.transform().parent() = &t1.transform();
    t3.transform().parent() = &t2.transform();

    t3.transform().parent() = &t0.transform();

    EXPECT_TRUE(*t1.parent() == t0);
    EXPECT_TRUE(*t2.parent() == t1);
    EXPECT_TRUE(*t3.parent() == t0);

    EXPECT_TRUE(*t1.transform().parent() == t0.transform());
    EXPECT_TRUE(*t2.transform().parent() == t1.transform());
    EXPECT_TRUE(*t3.transform().parent() == t0.transform());
  }

  {
    Object &o0 = Object::Create();
    Object &o1 = Object::Create();
    Object &o2 = Object::Create();
    Object &o3 = Object::Create();

    Transform &t0 = o0.transform();
    Transform &t1 = o1.transform();
    Transform &t2 = o2.transform();
    Transform &t3 = o3.transform();

    t1.transform().parent() = &t0.transform();
    t2.transform().parent() = &t1.transform();
    t3.transform().parent() = &t2.transform();

    t2.transform().parent() = &t0.transform();

    EXPECT_TRUE(*t1.parent() == t0);
    EXPECT_TRUE(*t2.parent() == t0);
    EXPECT_TRUE(*t3.parent() == t2);

    EXPECT_TRUE(*t1.transform().parent() == t0.transform());
    EXPECT_TRUE(*t2.transform().parent() == t0.transform());
    EXPECT_TRUE(*t3.transform().parent() == t2.transform());
  }

  {
    Object &o0 = Object::Create();
    Object &o1 = Object::Create();
    Object &o2 = Object::Create();
    Object &o3 = Object::Create();

    Transform &t0 = o0.transform();
    Transform &t1 = o1.transform();
    Transform &t2 = o2.transform();
    Transform &t3 = o3.transform();

    t1.transform().parent() = &t0.transform();
    t2.transform().parent() = &t1.transform();
    t3.transform().parent() = &t2.transform();

    bool failed = false;
    try {
      t2.transform().parent() = &t3.transform();
    } catch (std::exception &e) { failed = true; }

    EXPECT_TRUE(failed);
  }

  // Complex roots.
  {
    Object &o0 = Object::Create();
    Object &o1 = Object::Create();
    Object &o2 = Object::Create();
    Object &o3 = Object::Create();

    Transform &t0 = o0.transform();
    Transform &t1 = o1.transform();
    Transform &t2 = o2.transform();
    Transform &t3 = o3.transform();

    t1.parent() = &t0;
    t3.parent() = &t2;

    t2.root() = &t0;

    EXPECT_TRUE(*t1.parent() == t0);
    EXPECT_TRUE(*t2.parent() == t0);
    EXPECT_TRUE(*t3.parent() == t2);
  }

  {
    Object &o0 = Object::Create();
    Object &o1 = Object::Create();
    Object &o2 = Object::Create();
    Object &o3 = Object::Create();

    Transform &t0 = o0.transform();
    Transform &t1 = o1.transform();
    Transform &t2 = o2.transform();
    Transform &t3 = o3.transform();

    t1.parent() = &t0;
    t3.parent() = &t2;

    t3.root() = &t0;

    EXPECT_TRUE(*t1.parent() == t0);
    EXPECT_TRUE(*t2.parent() == t0);
    EXPECT_TRUE(*t3.parent() == t2);
  }

  {
    Object &o0 = Object::Create();
    Object &o1 = Object::Create();
    Object &o2 = Object::Create();
    Object &o3 = Object::Create();

    Transform &t0 = o0.transform();
    Transform &t1 = o1.transform();
    Transform &t2 = o2.transform();
    Transform &t3 = o3.transform();

    t1.parent() = &t0;
    t2.parent() = &t1;
    t3.parent() = &t2;

    bool failed = false;
    try {
      t2.root() = &t3;
    } catch (std::exception &e) { failed = true; }

    EXPECT_TRUE(failed);
  }

  {
    Object &o0 = Object::Create();
    Object &o1 = Object::Create();
    Object &o2 = Object::Create();
    Object &o3 = Object::Create();

    Transform &t0 = o0.transform();
    Transform &t1 = o1.transform();
    Transform &t2 = o2.transform();
    Transform &t3 = o3.transform();

    t1.transform().parent() = &t0.transform();
    t3.transform().parent() = &t2.transform();

    t2.transform().root() = &t0.transform();

    EXPECT_TRUE(*t1.parent() == t0);
    EXPECT_TRUE(*t2.parent() == t0);
    EXPECT_TRUE(*t3.parent() == t2);

    EXPECT_TRUE(*t1.transform().parent() == t0.transform());
    EXPECT_TRUE(*t2.transform().parent() == t0.transform());
    EXPECT_TRUE(*t3.transform().parent() == t2.transform());
  }

  {
    Object &o0 = Object::Create();
    Object &o1 = Object::Create();
    Object &o2 = Object::Create();
    Object &o3 = Object::Create();

    Transform &t0 = o0.transform();
    Transform &t1 = o1.transform();
    Transform &t2 = o2.transform();
    Transform &t3 = o3.transform();

    t1.transform().parent() = &t0.transform();
    t3.transform().parent() = &t2.transform();

    t3.transform().root() = &t0.transform();

    EXPECT_TRUE(*t1.parent() == t0);
    EXPECT_TRUE(*t2.parent() == t0);
    EXPECT_TRUE(*t3.parent() == t2);

    EXPECT_TRUE(*t1.transform().parent() == t0.transform());
    EXPECT_TRUE(*t2.transform().parent() == t0.transform());
    EXPECT_TRUE(*t3.transform().parent() == t2.transform());
  }

  {
    Object &o0 = Object::Create();
    Object &o1 = Object::Create();
    Object &o2 = Object::Create();
    Object &o3 = Object::Create();

    Transform &t0 = o0.transform();
    Transform &t1 = o1.transform();
    Transform &t2 = o2.transform();
    Transform &t3 = o3.transform();

    t1.transform().parent() = &t0.transform();
    t2.transform().parent() = &t1.transform();
    t3.transform().parent() = &t2.transform();

    bool failed = false;
    try {
      t2.transform().root() = &t3.transform();
    } catch (std::exception &e) { failed = true; }

    EXPECT_TRUE(failed);
  }

  // Sub-properties of parent and root.
  {
    Object &o0 = Object::Create();
    Object &o1 = Object::Create();
    Object &o2 = Object::Create();
    Object &o3 = Object::Create();

    Transform &t0 = o0.transform();
    Transform &t1 = o1.transform();
    Transform &t2 = o2.transform();
    Transform &t3 = o3.transform();

    t1.parent() = &t0;
    t2.parent() = &t1;
    t3.parent() = &t2;

    // TODO: Something like `trans.parent()->parent()` is forbidden because
    // clang does not allow the child class to have the same name with the parent class.
#if 0
    // Level 2.
    EXPECT_TRUE(*t2.parent()->parent() == t0);
    EXPECT_TRUE(*t2.parent()->root() == t0);

    EXPECT_TRUE(*t2.root()->parent() == *t0.parent());
    EXPECT_TRUE(*t2.root()->root() == t0);

    EXPECT_TRUE(*t3.parent()->parent() == t1);
    EXPECT_TRUE(*t3.parent()->root() == t0);

    EXPECT_TRUE(*t3.root()->parent() == *t0.parent());
    EXPECT_TRUE(*t3.root()->root() == t0);

    // Level 3.
    EXPECT_TRUE(*t2.parent()->parent()->parent() == *t0.parent());

    EXPECT_TRUE(*t2.parent()->root()->parent() == *t0.parent());
    EXPECT_TRUE(*t2.parent()->root()->root() == t0);

    EXPECT_TRUE(*t2.root()->root()->parent() == *t0.parent());
    EXPECT_TRUE(*t2.root()->root()->root() == t0);

    //
    EXPECT_TRUE(*t3.parent()->parent()->parent() == t0);

    EXPECT_TRUE(*t3.parent()->root()->parent() == *t0.parent());
    EXPECT_TRUE(*t3.parent()->root()->root() == t0);

    EXPECT_TRUE(*t3.root()->root()->parent() == *t0.parent());
    EXPECT_TRUE(*t3.root()->root()->root() == t0);
#endif
  }

  // Is root and is child of.
  {
    Object &o0 = Object::Create();
    Object &o1 = Object::Create();
    Object &o2 = Object::Create();
    Object &o3 = Object::Create();

    Transform &t0 = o0.transform();
    Transform &t1 = o1.transform();
    Transform &t2 = o2.transform();
    Transform &t3 = o3.transform();

    t1.parent() = &t0;
    t2.parent() = &t1;
    t3.parent() = &t2;

    EXPECT_TRUE(t0.IsRoot());
    EXPECT_FALSE(t1.IsRoot());
    EXPECT_FALSE(t2.IsRoot());
    EXPECT_FALSE(t3.IsRoot());

    EXPECT_TRUE(t1.IsChildOf(t0));
    EXPECT_TRUE(t2.IsChildOf(t0));
    EXPECT_TRUE(t2.IsChildOf(t1));
    EXPECT_TRUE(t3.IsChildOf(t0));
    EXPECT_TRUE(t3.IsChildOf(t1));
    EXPECT_TRUE(t3.IsChildOf(t2));

    EXPECT_FALSE(t0.IsChildOf(t0));
    EXPECT_FALSE(t0.IsChildOf(t1));
    EXPECT_FALSE(t0.IsChildOf(t2));
    EXPECT_FALSE(t0.IsChildOf(t3));
    EXPECT_FALSE(t1.IsChildOf(t1));
    EXPECT_FALSE(t1.IsChildOf(t2));
    EXPECT_FALSE(t1.IsChildOf(t3));
    EXPECT_FALSE(t2.IsChildOf(t2));
    EXPECT_FALSE(t2.IsChildOf(t3));
    EXPECT_FALSE(t3.IsChildOf(t3));
  }
}

TEST(Transform, Local) {
  auto expectV = [](const Vec3r &lhs, const Vec3r &rhs) {
    EXPECT_FLOAT_EQ(float(lhs.x()), float(rhs.x()));
    EXPECT_FLOAT_EQ(float(lhs.y()), float(rhs.y()));
    EXPECT_FLOAT_EQ(float(lhs.z()), float(rhs.z()));
  };

  auto expectQ = [](const Quatr &lhs, const Quatr &rhs) {
    EXPECT_FLOAT_EQ(float(lhs.w()), float(rhs.w()));
    EXPECT_FLOAT_EQ(float(lhs.x()), float(rhs.x()));
    EXPECT_FLOAT_EQ(float(lhs.y()), float(rhs.y()));
    EXPECT_FLOAT_EQ(float(lhs.z()), float(rhs.z()));
  };

  // Position, rotation, and scale.
  {
    Object &o = Object::Create();
    Transform &t = o.transform();

    expectV(t.localPosition(), {0, 0, 0});
    expectQ(t.localRotation(), {1, 0, 0, 0});
    expectV(t.localScale(), {1, 1, 1});

    t.localPosition() += Vec3r{1, 2, 3};
    expectV(t.localPosition(), {1, 2, 3});
  }

  // Up, down, forward, back, left, and right.
  {
    Object &o = Object::Create();
    Transform &t = o.transform();

    t.localRotation() = FromEulerAngles(Vec3r{30, 0, 0} * deg2Rad<Real>);
    expectV(t.localEulerAngles(), ToEulerAngles(static_cast<Quatr>(t.localRotation())));
    expectV(t.localEulerAngles(), ToEulerAngles(t.localRotation()));

    expectV(t.localUp(), {0, 0.8660254, 0.5});
    expectV(t.localDown(), {0, -0.8660254, -0.5});

    t.localEulerAngles() = {0_R, deg2Rad<Real> * 30, 0_R};
    expectV(t.localForward(), {-0.5, 0, -0.8660254});
    expectV(t.localBack(), {0.5, 0, 0.8660254});

    t.localEulerAngles() = {0_R, 0_R, deg2Rad<Real> * 30};
    expectV(t.localLeft(), {-0.8660254, -0.5, 0});
    expectV(t.localRight(), {0.8660254, 0.5, 0});
  }
}

TEST(Transform, World) {
  auto expectV = [](const Vec3r &lhs, const Vec3r &rhs) {
    EXPECT_FLOAT_EQ(float(lhs.x()), float(rhs.x()));
    EXPECT_FLOAT_EQ(float(lhs.y()), float(rhs.y()));
    EXPECT_FLOAT_EQ(float(lhs.z()), float(rhs.z()));
  };

  auto expectVRelaxed = [](const Vec3r &lhs, const Vec3r &rhs) {
    EXPECT_TRUE(std::abs(float(lhs.x()) - float(rhs.x())) < 5e-4);
    EXPECT_TRUE(std::abs(float(lhs.y()) - float(rhs.y())) < 5e-4);
    EXPECT_TRUE(std::abs(float(lhs.z()) - float(rhs.z())) < 5e-4);
  };

  auto expectQ = [](const Quatr &lhs, const Quatr &rhs) {
    EXPECT_FLOAT_EQ(float(lhs.w()), float(rhs.w()));
    EXPECT_FLOAT_EQ(float(lhs.x()), float(rhs.x()));
    EXPECT_FLOAT_EQ(float(lhs.y()), float(rhs.y()));
    EXPECT_FLOAT_EQ(float(lhs.z()), float(rhs.z()));
  };

  Object &oGrandPare = Object::Create();
  Transform &tGrandPare = oGrandPare.transform();
  tGrandPare.localPosition() = {1.1F, 2.4F, 3.8F};
  tGrandPare.localRotation() = {0.2F, 0.5F, 0.7F, 0.6F};
  tGrandPare.localScale() = {0.5F, 0.8F, 0.7F};

  Object &oPare = Object::Create();
  oPare.parent() = &oGrandPare;
  Transform &tPare = oPare.transform();
  tPare.localPosition() = {4.5F, 16.6F, 14.7F};
  tPare.localRotation() = {0.5F, 0.3F, 0.9F, 0.8F};
  tPare.localScale() = {2.0F, 1.3F, 1.8F};

  // Global position, rotation, and 6 directions.
  {
    Object &o = Object::Create();
    o.parent() = &oPare;
    Transform &t = o.transform();
    t.localPosition() = {-3.14F, -1.59F, -2.6F};
    t.localRotation() = {0.31F, 0.415F, 0.9F, 0.26F};
    t.localScale() = {3.1F, 4.15F, 9.26F};

    // Get.
    Vec3r lp = t.localPosition();
    Vec3r wp = t.position();
    Quatr lr = t.localRotation();
    Quatr wr = t.rotation();
    Vec3r ls = t.localScale();

    // Set.
    expectV(lp, {-3.14000010, -1.59000003, -2.59999990});
    expectV(wp, {10.27433395, 9.59436226, 7.78195524});
    expectQ(lr, {0.28959003, 0.38767698, 0.84074527, 0.24288197});
    expectQ(wr, {-0.62618756, -0.60719037, -0.48601389, 0.05476814});
    expectV(ls, {3.09999990, 4.15000010, 9.26000023});

    // Set.
    auto expectPares = [&] {
      expectV(tGrandPare.position(), {1.1F, 2.4F, 3.8F});
      expectQ(tGrandPare.rotation(), Quatr{0.2F, 0.5F, 0.7F, 0.6F}.normalized());
      expectV(tGrandPare.localScale(), {0.5F, 0.8F, 0.7F});

      expectV(tPare.localPosition(), {4.5F, 16.6F, 14.7F});
      expectQ(tPare.localRotation(), Quatr{0.5F, 0.3F, 0.9F, 0.8F}.normalized());
      expectV(tPare.localScale(), {2.0F, 1.3F, 1.8F});
    };

    expectPares();

    t.position() = {1.0F, 2.0F, 3.0F};
    expectPares();
    expectVRelaxed(t.position(), {1, 2, 3});

    t.rotation() = {4.0F, 8.0F, 7.0F, 6.0F};
    expectPares();
    expectVRelaxed(t.position(), {1, 2, 3});
    expectQ(t.rotation(), Quatr{4, 8, 7, 6}.normalized());

    t.up() = {5.5F, 3.33F, 13.131313F};
    expectPares();
    expectVRelaxed(t.position(), {1, 2, 3});
    expectVRelaxed(t.up(), Vec3r{5.5F, 3.33F, 13.131313F}.normalized());

    t.down() = {5.5F, 3.33F, 13.131313F};
    expectPares();
    expectVRelaxed(t.position(), {1, 2, 3});
    expectVRelaxed(t.down(), Vec3r{5.5F, 3.33F, 13.131313F}.normalized());

    t.forward() = {5.5F, 3.33F, 13.131313F};
    expectPares();
    expectVRelaxed(t.position(), {1, 2, 3});
    expectVRelaxed(t.forward(), Vec3r{5.5F, 3.33F, 13.131313F}.normalized());

    t.back() = {5.5F, 3.33F, 13.131313F};
    expectPares();
    expectVRelaxed(t.position(), {1, 2, 3});
    expectVRelaxed(t.back(), Vec3r{5.5F, 3.33F, 13.131313F}.normalized());

    t.left() = {5.5F, 3.33F, 13.131313F};
    expectPares();
    expectVRelaxed(t.position(), {1, 2, 3});
    expectVRelaxed(t.left(), Vec3r{5.5F, 3.33F, 13.131313F}.normalized());

    t.right() = {5.5F, 3.33F, 13.131313F};
    expectPares();
    expectVRelaxed(t.position(), {1, 2, 3});
    expectVRelaxed(t.right(), Vec3r{5.5F, 3.33F, 13.131313F}.normalized());
  }

  // Transform point, vector, and direction.
  {
    Object &o = Object::Create();
    o.parent() = &oPare;
    Transform &t = o.transform();
    t.localPosition() = {-3.14F, -1.59F, -2.6F};
    t.localRotation() = {0.31F, 0.415F, 0.9F, 0.26F};
    t.localScale() = {3.1F, 4.15F, 9.26F};

    Vec3r p = t.TransformPoint(10.0F, 14.0F, 23.0F);
    Vec3r v = t.TransformVector(10.0F, 14.0F, 23.0F);
    Vec3r d = t.TransformDirection(10.0F, 14.0F, 23.0F);

    expectV(p, {169.87333679, -136.12190247, 27.38156128});
    expectV(v, {159.59899902, -145.71626282, 19.59960556});
    expectV(d, {26.90869904, -9.90520096, -1.67601967});
  }

  // Translate, rotate, and rotateAround.
  {
    tGrandPare.localScale() = {2.0F, 2.0F, 2.0F};
    tPare.localScale() = {3.0F, 3.0F, 3.0F};

    // Translate at self space.
    {
      Object &o = Object::Create();
      o.parent() = &oPare;
      Transform &t = o.transform();
      t.localPosition() = {-3.14F, -1.59F, -2.6F};
      t.localRotation() = {0.31F, 0.415F, 0.9F, 0.26F};
      t.localScale() = {3.1F, 4.15F, 9.26F};

      t.Translate(10, 14, 23);

      Vec3r lp = t.localPosition();
      Vec3r wp = t.position();
      Quatr lr = t.localRotation();
      Quatr wr = t.rotation();
      Vec3r ls = t.localScale();
      Vec3r ws = t.lossyScale();

      expectV(lp, {-0.24483609, 1.79241288, -4.35896873});
      expectV(wp, {45.07867050, 14.08513069, 4.34870148});
      expectQ(lr, {0.28959003, 0.38767698, 0.84074527, 0.24288197});
      expectQ(wr, {-0.62618756, -0.60719037, -0.48601389, 0.05476814});
      expectV(ls, {3.09999990, 4.15000010, 9.26000023});
      expectV(ws, {18.60000610, 24.90000343, 55.56002426});
    }

    // Translate at world space.
    {
      Object &o = Object::Create();
      o.parent() = &oPare;
      Transform &t = o.transform();
      t.localPosition() = {-3.14F, -1.59F, -2.6F};
      t.localRotation() = {0.31F, 0.415F, 0.9F, 0.26F};
      t.localScale() = {3.1F, 4.15F, 9.26F};

      t.Translate(10, 14, 23, Space::World);

      Vec3r lp = t.localPosition();
      Vec3r wp = t.position();
      Quatr lr = t.localRotation();
      Quatr wr = t.rotation();
      Vec3r ls = t.localScale();
      Vec3r ws = t.lossyScale();

      expectV(lp, {-1.83482933, 0.24429655, 1.62475491});
      expectV(wp, {28.16996002, 37.99033356, 29.02473068});
      expectQ(lr, {0.28959003, 0.38767698, 0.84074527, 0.24288197});
      expectQ(wr, {-0.62618756, -0.60719037, -0.48601389, 0.05476814});
      expectV(ls, {3.09999990, 4.15000010, 9.26000023});
      expectV(ws, {18.60000610, 24.90000343, 55.56002426});
    }

    // Rotate at self space.
    {
      Object &o = Object::Create();
      o.parent() = &oPare;
      Transform &t = o.transform();
      t.localPosition() = {-3.14F, -1.59F, -2.6F};
      t.localRotation() = {0.31F, 0.415F, 0.9F, 0.26F};
      t.localScale() = {3.1F, 4.15F, 9.26F};

      t.Rotate(deg2Rad<float> * 30.0F, deg2Rad<float> * 60.0F, deg2Rad<float> * 70.0F);

      Vec3r lp = t.localPosition();
      Vec3r wp = t.position();
      Quatr lr = t.localRotation();
      Quatr wr = t.rotation();
      Vec3r ls = t.localScale();
      Vec3r ws = t.lossyScale();

      expectV(lp, {-3.14000010, -1.59000003, -2.59999990});
      expectV(wp, {18.16996574, 23.99033165, 6.02472687});
      expectQ(lr, {-0.27534989, 0.45433092, 0.62271023, 0.57445449});
      expectQ(wr, {-0.29799291, -0.61302638, -0.47549367, -0.55615163});
      expectV(ls, {3.09999990, 4.15000010, 9.26000023});
      expectV(ws, {18.60000610, 24.90001297, 55.56002045});
    }

    // Rotate at world space.
    {
      Object &o = Object::Create();
      o.parent() = &oPare;
      Transform &t = o.transform();
      t.localPosition() = {-3.14F, -1.59F, -2.6F};
      t.localRotation() = {0.31F, 0.415F, 0.9F, 0.26F};
      t.localScale() = {3.1F, 4.15F, 9.26F};

      t.Rotate(deg2Rad<float> * 30.0F, deg2Rad<float> * 60.0F, deg2Rad<float> * 70.0F, Space::World);

      Vec3r lp = t.localPosition();
      Vec3r wp = t.position();
      Quatr lr = t.localRotation();
      Quatr wr = t.rotation();
      Vec3r ls = t.localScale();
      Vec3r ws = t.lossyScale();

      expectV(lp, {-3.14000010, -1.59000003, -2.59999990});
      expectV(wp, {18.16996574, 23.99033165, 6.02472687});
      expectQ(lr, {0.08198463, -0.26261568, 0.94486064, 0.17762274});
      expectQ(wr, {-0.29799297, -0.19226539, -0.91919744, 0.17120233});
      expectV(ls, {3.09999990, 4.15000010, 9.26000023});
      expectV(ws, {18.60000038, 24.90000343, 55.56000519});
    }

    // RotateAround.
    {
      Object &o = Object::Create();
      o.parent() = &oPare;
      Transform &t = o.transform();
      t.localPosition() = {-3.14F, -1.59F, -2.6F};
      t.localRotation() = {0.31F, 0.415F, 0.9F, 0.26F};
      t.localScale() = {3.1F, 4.15F, 9.26F};

      t.RotateAround({1, 2, 3}, {4, -5, 6}, deg2Rad<float> * 15.0F);

      Vec3r lp = t.localPosition();
      Vec3r wp = t.position();
      Quatr lr = t.localRotation();
      Quatr wr = t.rotation();
      Vec3r ls = t.localScale();
      Vec3r ws = t.lossyScale();

      expectV(lp, {-3.24329662, -2.32918715, -1.64052534});
      expectV(wp, {13.20625687, 25.97391891, 10.98685646});
      expectQ(lr, {0.22552405, 0.40670648, 0.81242532, 0.35170150});
      expectQ(wr, {-0.62573791, -0.59995055, -0.49273366, -0.07566389});
      expectV(ls, {3.09999990, 4.15000010, 9.26000023});
      expectV(ws, {18.60000610, 24.90000343, 55.56003189});
    }
  }
}

TEST(Transform, LossyScale) {
  // TODO: Support robust lossyScale().
}

} // namespace ARIA
