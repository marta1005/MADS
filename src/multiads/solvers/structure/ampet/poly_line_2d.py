# ruff: noqa: N802, N803, N806

import math


class PolyLine2D:
    def __init__(self, curve: list[list[float]]) -> None:
        self.nSegments = len(curve) - 1

        self.x: list[float] = [[0.0] for _ in range(self.nSegments + 1)]
        self.y: list[float] = [[0.0] for _ in range(self.nSegments + 1)]

        self.length = 0.0

        self.x[0] = curve[0][0]
        self.y[0] = curve[0][1]

        self.minX = self.x[0]
        self.maxX = self.x[0]
        self.minY = self.y[0]
        self.maxY = self.y[0]

        self.area = 0.0

        for iPoint in range(1, self.nSegments + 1):
            self.x[iPoint] = curve[iPoint][0]
            self.y[iPoint] = curve[iPoint][1]

            dx = self.x[iPoint] - self.x[iPoint - 1]
            dy = self.y[iPoint] - self.y[iPoint - 1]

            self.length += math.sqrt(dx * dx + dy * dy)

            self.minX = min(self.minX, self.x[iPoint])
            self.maxX = max(self.maxX, self.x[iPoint])
            self.minY = min(self.minY, self.y[iPoint])
            self.maxY = max(self.maxY, self.y[iPoint])

            self.area += 0.5 * (self.y[iPoint] + self.y[iPoint - 1]) * dx

        self.structuralProperties()

    def rescaleAndMove(
        self,
        xFactor: float = 1.0,
        yFactor: float = 1.0,
        xOrigin: float = 0.0,
        yOrigin: float = 0.0,
    ) -> "PolyLine2D":
        solutionCurve = [[0.0, 0.0] for _ in range(self.nSegments + 1)]

        for iPoint in range(self.nSegments + 1):
            solutionCurve[iPoint][0] = self.x[iPoint] * xFactor - xOrigin
            solutionCurve[iPoint][1] = self.y[iPoint] * yFactor - yOrigin

        return PolyLine2D(solutionCurve)

    def rotate(
        self,
        angle: float = 0.0,
        xOrigin: float = 0.0,
        yOrigin: float = 0.0,
    ) -> "PolyLine2D":
        solutionCurve = [[0.0, 0.0] for _ in range(self.nSegments + 1)]

        for iPoint in range(self.nSegments + 1):
            dx = self.x[iPoint] - xOrigin
            dy = self.y[iPoint] - yOrigin

            solutionCurve[iPoint][0] = (
                xOrigin + math.cos(angle) * dx - math.sin(angle) * dy
            )
            solutionCurve[iPoint][1] = (
                yOrigin + math.cos(angle) * dy + math.sin(angle) * dx
            )

        return PolyLine2D(solutionCurve)

    def breakNParts(self, nParts: int) -> list["PolyLine2D"]:
        lStep = self.length / nParts

        solution = [PolyLine2D([[0.0, 0.0], [0.0, 1.0]]) for _ in range(nParts)]
        iSolution = 0
        l0 = 0.0
        x1 = self.x[0]
        y1 = self.y[0]
        lCut = lStep

        tSolution = [[x1, y1]]

        epsL = lStep * 1e-10

        for iPoint in range(1, self.nSegments + 1):
            x2 = self.x[iPoint]
            y2 = self.y[iPoint]

            dx = x2 - x1
            dy = y2 - y1

            l1 = math.sqrt(dx * dx + dy * dy)
            l = l0 + l1  # noqa: E741

            while l > (lCut - epsL):
                dl = lCut - l0
                x3 = x1 + (x2 - x1) / l1 * dl
                y3 = y1 + (y2 - y1) / l1 * dl

                tSolution += [[x3, y3]]

                solution[iSolution] = PolyLine2D(tSolution)
                iSolution += 1
                l1 -= dl
                l0 = lCut
                lCut += lStep
                tSolution = [[x3, y3]]
                if l > (lCut - epsL):
                    x1 = x3
                    y1 = y3

            if abs(l - lCut) < lStep * 1e-10:
                tSolution += [[x2, y2]]
                solution[iSolution] = PolyLine2D(tSolution)
                tSolution = [[x2, y2]]
                iSolution += 1
                lCut += lStep

            else:
                tSolution += [[x2, y2]]

            x1 = x2
            y1 = y2
            l0 = l

        return solution

    def breakByXPosition(self, xCut: float) -> list["PolyLine2D"]:
        solution = [PolyLine2D([[0.0, 0.0], [0.0, 1.0]]) for _ in range(2)]

        iSolution = 0
        x1 = self.x[0]
        y1 = self.y[0]

        tSolution = [[x1, y1]]

        for iPoint in range(1, self.nSegments + 1):
            x2 = self.x[iPoint]
            y2 = self.y[iPoint]

            if ((x2 > xCut) != (x1 > xCut)) and (iSolution == 0):
                x3 = xCut
                y3 = y1 + (y2 - y1) / (x2 - x1) * (x3 - x1)
                tSolution += [[x3, y3]]
                solution[iSolution] = PolyLine2D(tSolution)
                tSolution = [[x3, y3], [x2, y2]]
                iSolution = 1
            elif (x2 == xCut) and (iSolution == 0):
                tSolution += [[x2, y2]]
                solution[iSolution] = PolyLine2D(tSolution)
                tSolution = [[x2, y2]]
                iSolution = 1
            else:
                tSolution += [[x2, y2]]

            x1 = x2
            y1 = y2

        solution[iSolution] = PolyLine2D(tSolution)

        return solution

    def tangentInL(self, lPosition: float) -> list[float]:
        solution = [0.0, 0.0]

        l0 = 0.0
        x1 = self.x[0]
        y1 = self.y[0]

        epsL = self.length * 1.0e-10

        for iPoint in range(1, self.nSegments + 1):
            x2 = self.x[iPoint]
            y2 = self.y[iPoint]

            dx = x2 - x1
            dy = y2 - y1

            dl2 = math.sqrt(dx * dx + dy * dy)
            l = l0 + dl2  # noqa: E741

            if (l > lPosition) and (l0 < lPosition):
                solution = [(x2 - x1) / dl2, (y2 - y1) / dl2]

            elif abs(l0 - lPosition) < epsL:
                if iPoint > 1:
                    dxdl = (dl1 * dl1 * (x2 - x1) + dl2 * dl2 * (x1 - x0)) / (
                        dl1 * dl2 * (dl1 + dl2)
                    )
                    dydl = (dl1 * dl1 * (y2 - y1) + dl2 * dl2 * (y1 - y0)) / (
                        dl1 * dl2 * (dl1 + dl2)
                    )
                    factor = math.sqrt(dxdl * dxdl + dydl * dydl)
                    solution = [dxdl / factor, dydl / factor]
                else:
                    solution = [(x2 - x1) / dl2, (y2 - y1) / dl2]

            elif (abs(l - lPosition) < epsL) and (iPoint == self.nSegments):
                solution = [(x2 - x1) / dl2, (y2 - y1) / dl2]

            x0 = x1
            y0 = y1
            dl1 = dl2

            x1 = x2
            y1 = y2
            l0 = l

        return solution

    def radiusInPosition(self, position: int) -> float:
        if (position <= 0) or (position >= self.nSegments):
            radius = 0.0

        else:
            x0 = self.x[position - 1]
            x1 = self.x[position]
            x2 = self.x[position + 1]

            y0 = self.y[position - 1]
            y1 = self.y[position]
            y2 = self.y[position + 1]

            l1 = math.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0))
            l2 = math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

            x3 = 0.5 * (x0 + x1)
            y3 = 0.5 * (y0 + y1)
            ux = (y1 - y0) / l1
            uy = (x0 - x1) / l1

            x4 = 0.5 * (x2 + x1)
            y4 = 0.5 * (y2 + y1)
            vx = (y2 - y1) / l2
            vy = (x1 - x2) / l2

            det = uy * vx - ux * vy

            lambd = (-vy * (x4 - x3) + vx * (y4 - y3)) / det

            xc = x3 + lambd * ux
            yc = y3 + lambd * uy

            radius = math.sqrt((xc - x1) * (xc - x1) + (yc - y1) * (yc - y1))

        return radius

    def reverse(self) -> "PolyLine2D":
        solution = PolyLine2D([[0.0, 0.0], [0.0, 1.0]])

        solution.nSegments = self.nSegments
        solution.x = self.x[::-1]
        solution.y = self.y[::-1]
        solution.length = self.length
        solution.minX = self.minX
        solution.maxX = self.maxX
        solution.minY = self.minY
        solution.maxY = self.maxY
        solution.area = -self.area

        solution.structuralProperties()

        return solution

    def append(self, poly2: "PolyLine2D") -> "PolyLine2D":
        eps = 1.0e-10

        xF1 = self.x[self.nSegments]
        yF1 = self.y[self.nSegments]

        tanF1 = self.tangentInL(self.length)

        x02 = poly2.x[0]
        y02 = poly2.y[0]

        tan02 = poly2.tangentInL(0.0)

        solution = [[self.x[0], self.y[0]]]

        for iPoint in range(1, self.nSegments):
            solution += [[self.x[iPoint], self.y[iPoint]]]

        if (abs(xF1 - x02) < eps) and (abs(yF1 - y02) < eps):
            if (abs(tanF1[0] - tan02[0]) > eps) or (abs(tanF1[1] - tan02[1]) > eps):
                solution += [[xF1, yF1]]

        else:
            solution += [[xF1, yF1]]
            solution += [[x02, y02]]

        for iPoint in range(1, poly2.nSegments + 1):
            solution += [[poly2.x[iPoint], poly2.y[iPoint]]]

        return PolyLine2D(solution)

    def structuralProperties(self) -> None:
        x1 = self.x[0]
        y1 = self.y[0]

        xA = 0.0
        yA = 0.0

        Axxt = 0.0
        Axxt3 = 0.0
        Axyt = 0.0
        Axyt3 = 0.0
        Ayyt = 0.0
        Ayyt3 = 0.0

        rhoL = 0.0
        rhoLL = 0.0
        rhoLX = 0.0
        rhoLY = 0.0

        l0 = 0.0
        xA0 = 0.0
        yA0 = 0.0

        for iPoint in range(1, self.nSegments + 1):
            x2 = self.x[iPoint]
            y2 = self.y[iPoint]

            dx = x2 - x1
            dy = y2 - y1

            lt = math.sqrt(dx * dx + dy * dy)

            dAx = 0.5 * (x1 + x2) * lt
            dAy = 0.5 * (y1 + y2) * lt

            xA = xA + dAx
            yA = yA + dAy

            Axxt = Axxt + (x1 * x1 + x2 * x1 + x2 * x2) * lt / 3.0
            Axxt3 = Axxt3 + dy * dy / (12.0 * lt)

            Axyt = Axyt + (2.0 * x1 * y1 + x1 * y2 + x2 * y1 + 2.0 * x2 * y2) * lt / 6.0
            Axyt3 = Axyt3 - dx * dy / (12.0 * lt)

            Ayyt = Ayyt + (y1 * y1 + y2 * y1 + y2 * y2) * lt / 3.0
            Ayyt3 = Ayyt3 + dx * dx / (12.0 * lt)

            dRhoL = x2 * y1 - x1 * y2
            rhoL = rhoL + dRhoL
            rhoLL = rhoLL + dRhoL * (l0 + 0.5 * lt)
            rhoLX = rhoLX + dRhoL * (xA0 + 0.5 * dAx)
            rhoLY = rhoLY + dRhoL * (yA0 + 0.5 * dAy)

            x1 = x2
            y1 = y2
            l0 = l0 + lt
            xA0 = xA
            yA0 = yA

        self.inertiaVector = [
            self.length,
            xA,
            yA,
            [Axxt, Axxt3],
            [Axyt, Axyt3],
            [Ayyt, Ayyt3],
        ]
        self.shearProperties = [rhoL, rhoLL, rhoLX, rhoLY]

    def getInertiaVector(self, thickness: float) -> list[float]:
        solution = [0.0 for _ in range(6)]

        for iVector in range(3):
            solution[iVector] = self.inertiaVector[iVector] * thickness
            solution[iVector + 3] = (
                self.inertiaVector[iVector + 3][0]
                + self.inertiaVector[iVector + 3][1] * thickness * thickness
            ) * thickness

        return solution

    def getPoint(self, lCut: float) -> list[float]:
        x1 = self.x[0]
        y1 = self.y[0]
        l0 = 0.0

        for iPoint in range(1, self.nSegments + 1):
            x2 = self.x[iPoint]
            y2 = self.y[iPoint]

            dx = x2 - x1
            dy = y2 - y1

            l1 = math.sqrt(dx * dx + dy * dy)
            l = l0 + l1  # noqa: E741

            if l > lCut:
                x3 = x1 + (x2 - x1) / l1 * (lCut - l0)
                y3 = y1 + (y2 - y1) / l1 * (lCut - l0)

                return [x3, y3]

            if l == lCut:
                return [x2, y2]

            solution = [x2, y2]

            x1 = x2
            y1 = y2
            l0 = l

        return solution
