/*
Trabaja con imagenes solo.
Usa el c�digo de Rostro 2 pero mejorado en el tema de los ojos y limpio
*/

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#define PI 3.14159265
#endif

#include <stdio.h>
#include <tchar.h>
#include <time.h>
#include <opencv2\opencv.hpp>
#include <SDKDDKVer.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat rotate(Mat src, double angle)
{
	Mat dst;
	Point2f pt(src.cols / 2., src.rows / 2.);
	Mat r = getRotationMatrix2D(pt, angle, 1.0);
	warpAffine(src, dst, r, Size(src.cols, src.rows));
	return dst;
}

struct caracteristicas {
	int x;		//centro.x
	int y;		//centro.y
	int ancho;
	int alto;
};
struct cara {
	caracteristicas CojoD;
	caracteristicas CojoI;
	caracteristicas Cnariz;
	caracteristicas Cboca;
	int perfil;			//0: cara frontal	1: perfil izq	2:perfil derecho
	int ang;			//angulo sobre eje z del rostro
};

int main()
{
	/****Variables auxiliares para el desarrollo y comprobaci�n del codigo*****/
	int detectCara = 0; int detectOjos = 1; int detectNariz = 1; int detectBoca = 1; int procesIma = 1; int ImagenModify = 0; //0:nada/1:empeque�er/2:agrandar
	int guardarIma = 1; int Reglas = 1;

	/*************************************************************************************************************/
	/****************Deteectores Viola-Jones de Caracter�sticas***************************************************/

	CascadeClassifier detector_cara;
	CascadeClassifier detector_ojos, detector_ojosIzq, detector_ojosDer;
	CascadeClassifier detector_boca, detector_boca1;
	CascadeClassifier detector_nariz, detector_nariz2, detector_nariz3;

	if (detectCara) {
		if (!detector_cara.load("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml"))
		{
			cout << "No se puede abrir el clasificador de cara." << endl;
			system("PAUSE"); //Espera entrada de usuario por teclado
			return -1;
		}
	}
	if (detectOjos) {
		if (!detector_ojos.load("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_eye.xml"))
		{
			cout << "No se puede abrir el clasificador de ojos." << endl;
			system("PAUSE");
			return -1;
		}
		if (!detector_ojosIzq.load("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_lefteye_2splits.xml"))
		{
			cout << "No se puede abrir el clasificador de ojos izquierdos." << endl;
			system("PAUSE");
			return -1;
		}
		if (!detector_ojosDer.load("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_righteye_2splits.xml"))
		{
			cout << "No se puede abrir el clasificador de ojos derechos." << endl;
			system("PAUSE");
			return -1;
		}
	}
	if (detectNariz) {
		if (!detector_nariz.load("C:\\Users\\AmandaGM\\Desktop\\TFM\\xml\\nariz.xml"))
		{
			cout << "No se puede abrir el clasificador de nariz." << endl;
			system("PAUSE");
			return -1;
		}
		if (!detector_nariz2.load("C:\\Users\\AmandaGM\\Desktop\\TFM\\xml\\FaceFeaturesDetectors\\Nariz.xml"))
		{
			cout << "No se puede abrir el clasificador de nariz2." << endl;
			system("PAUSE");
			return -1;
		}
		if (!detector_nariz3.load("C:\\Users\\AmandaGM\\Desktop\\TFM\\xml\\FaceFeaturesDetectors\\Nariz_nuevo_20stages.xml"))
		{
			cout << "No se puede abrir el clasificador de nariz3." << endl;
			system("PAUSE");
			return -1;
		}
	}
	if (detectBoca) {
		if (!detector_boca.load("C:\\Users\\AmandaGM\\Desktop\\TFM\\xml\\boca.xml"))
		{
			cout << "No se puede abrir el clasificador de boca." << endl;
			system("PAUSE");
			return -1;
		}
		if (!detector_boca1.load("C:\\Users\\AmandaGM\\Desktop\\TFM\\xml\\FaceFeaturesDetectors\\Mouth.xml"))
		{
			cout << "No se puede abrir el clasificador de boca1." << endl;
			system("PAUSE");
			return -1;
		}
	}

	/*************GUARDAR IMAGENES************************************************************************************************/
	int indG = 0; vector<string> index(1000); vector<int> compression_params;
	if (guardarIma)
	{
		compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
		compression_params.push_back(9);
	}

	/****************************************************************************************************************/
	Mat imagen;
	int ind_read = 0;	//Indice para leer imagen

	while (1)
	{
		/************************************Leer la imagen**********************************************************/
		ind_read++;
		String ImaNombre = "C:\\Users\\AmandaGM\\Desktop\\TFM\\Rostros\\Imagen (" + std::to_string(ind_read) + ").png";
		cout << endl << "Imagen a leer es : " << ImaNombre << endl;
		imagen = imread(ImaNombre, CV_LOAD_IMAGE_COLOR);
		if (!imagen.data)										//Comprobar que se ha le�do la im�gen
		{
			cout << "No se puede abrir la im�gen" << std::endl;
			system("PAUSE");									//Espera entrada de usuario por teclado
			return -1;
		}

		/******************************************************************************************
		EMPIEZA EL PROGRAMA
		******************************************************************************************/
		int alfa = -70; int dist_ojos = 0; int perfil = 0;  //0: frontal	1:izquierda		2:derecha
		float factor = 0.0f;  ImagenModify = 0;
		Mat im_gris, im_equa, im_bin;
		Mat Rim_imagen, Rim_gris, Rim_equa;		//Variables para la imagen de la cara
		Mat im_rotate = imagen;

		cvtColor(imagen, im_gris, CV_BGR2GRAY); //Guardamos en gris la imagen convertida a tonos grises
		equalizeHist(im_gris, im_equa);			//Ecualizamos la imagen para estandarizar el contraste

		if (detectCara) {
			CascadeClassifier detector_cara;
			vector<Rect> vect_cara;					//Vector para guardar las caracter�sticas de los rostros encontrados
			detector_cara.detectMultiScale(im_equa, vect_cara, 1.2, 3, 0, Size(60, 60));

			while (vect_cara.empty() && alfa < 55) {
				alfa = alfa + 15;
				im_rotate = rotate(im_equa, alfa);
				detector_cara.detectMultiScale(im_rotate, vect_cara, 1.2, 3, 0, Size(60, 60));
			}
			for (Rect rect : vect_cara)				
			{
				if (alfa != -70)
				{
					imagen = rotate(imagen, alfa); im_gris = rotate(im_gris, alfa); im_equa = rotate(im_equa, alfa);
				}
				Rim_gris = im_gris(rect);
				if (Rim_gris.size().width > 150) {
					Rim_equa = im_equa(rect);
					Rim_imagen = imagen(rect);

					rectangle(imagen,
						Point(rect.x, rect.y),
						Point(rect.x + rect.width, rect.y + rect.height),
						CV_RGB(0, 255, 0), 2);
				}
				else { cout << "DetectCara: Imagen reconocida muy peque�a" << endl; break; }
			}
		}
		else {
			Rim_gris = im_gris;
			Rim_equa = im_equa;
			Rim_imagen = imagen;
		}

		/*************Representaciones del rostro encontrado********************/
		if (procesIma) {
			factor = 400.0f / Rim_imagen.size().width;
		
			resize(Rim_imagen, Rim_imagen, Size(int(Rim_gris.cols * factor), int(Rim_gris.rows * factor)));
			resize(Rim_gris, Rim_gris, Size(int(Rim_gris.cols * factor), int(Rim_gris.rows * factor)));
			resize(Rim_equa, Rim_equa, Size(int(Rim_equa.cols * factor), int(Rim_equa.rows * factor)));
		// 	resize(imagen, imagen, Size(int(imagen.cols * factor), int(imagen.rows * factor)));
		}

		/*************Si hay cara, se buscan otras caracteristicas**************/
		Mat im_ojos_gris, im_ojos_equa, im_ojos_bin, im_ojos_binE, im_ojos_thr;
		Mat im_boca_gris, im_boca_equa;
		Mat im_gris_nariz, im_equa_nariz;
		vector<caracteristicas> oj, ojD, ojI, bo, na;

		if (detectOjos) {
			int indO = 1; int ho = 0;
			vector<Rect> vect_ojos, vect_ojosIzq, vect_ojosDer, vect_ojosE, vect_ojosIzqE, vect_ojosDerE;		//Vector para guardar las caracter�sticas de los ojos encontrados
			
			ho = Rim_gris.size().height * 3 / 16;
			im_ojos_gris = Rim_gris(Rect(0, Rim_gris.size().height * 3 / 16, Rim_gris.size().width, Rim_gris.size().height * 6 / 16));	namedWindow("O", WINDOW_AUTOSIZE);	moveWindow("O", 500, 0); imshow("O", im_ojos_gris);
			im_ojos_equa = Rim_equa(Rect(0, Rim_equa.size().height * 3 / 16, Rim_equa.size().width, Rim_equa.size().height * 6 / 16));	namedWindow("O_eq", WINDOW_AUTOSIZE);	moveWindow("O_eq", 500, 250); imshow("O_eq", im_ojos_equa);
			//	bilateralFilter(im_ojos_gris, im_ojos_bin, 15, 280, 80);	namedWindow("O_bin", WINDOW_AUTOSIZE);	moveWindow("O_bin", 900, 600); imshow("O_bin", im_ojos_bin);
			bilateralFilter(im_ojos_equa, im_ojos_binE, 15, 280, 80);	namedWindow("O_binE", WINDOW_AUTOSIZE);	moveWindow("O_binE", 1200, 600); imshow("O_binE", im_ojos_binE);
			cv::threshold(im_ojos_gris, im_ojos_thr, 0.65*sum(im_ojos_gris)[0] / (im_ojos_gris.cols*im_ojos_gris.rows), 255, CV_THRESH_BINARY); namedWindow("O2", WINDOW_AUTOSIZE);	moveWindow("O2", 500, 500); imshow("O2", im_ojos_thr);

			detector_ojos.detectMultiScale(im_ojos_gris, vect_ojos, 1.2, 3, 0, Size(60, 60), Size(120, 120));
			detector_ojosIzq.detectMultiScale(im_ojos_gris, vect_ojosIzq, 1.2, 3, 0, Size(60, 60), Size(120, 120));
			detector_ojosDer.detectMultiScale(im_ojos_gris, vect_ojosDer, 1.2, 3, 0, Size(60, 60), Size(120, 120));
			detector_ojos.detectMultiScale(im_ojos_equa, vect_ojosE, 1.2, 3, 0, Size(60, 60), Size(120, 120));
			detector_ojosIzq.detectMultiScale(im_ojos_equa, vect_ojosIzqE, 1.2, 3, 0, Size(60, 60), Size(120, 120));
			detector_ojosDer.detectMultiScale(im_ojos_equa, vect_ojosDerE, 1.2, 3, 0, Size(60, 60), Size(120, 120));

			if (vect_ojos.empty())
			{
				detector_ojos.detectMultiScale(im_ojos_binE, vect_ojos, 1.2, 3, 0, Size(60, 60));
			}
			/*		if (vect_ojosIzq.empty())
			{
			cout << "Oso bilateral Izq" << endl;
			detector_ojosIzq.detectMultiScale(im_ojos_binE, vect_ojosIzq, 1.2, 3, 0, Size(60, 60));

			}
			if (vect_ojosDer.empty())
			{
			cout << "Oso bilateral Der" << endl;
			detector_ojosDer.detectMultiScale(im_ojos_binE, vect_ojosDer, 1.2, 3, 0, Size(60, 60));
			}*/


			/*else if (ImagenModify == 1)
			{
			cout << "imagen war gross" << endl;
			h = h ;
			}*/

			for (Rect rectOjos : vect_ojos)
			{
				oj.push_back({ rectOjos.x + rectOjos.width / 2,  rectOjos.y + ho + rectOjos.height / 2, rectOjos.width, rectOjos.height});
				//	circle(imagen, Point(rectOjos.x + rectOjos.width / 2, rectOjos.y+ h + rectOjos.height / 2), (sqrt((rectOjos.width / 2)*(rectOjos.width / 2) + (rectOjos.height / 2) ^ 2)), Scalar(255, 0, 0), 3);
			}
			for (Rect rectOjos : vect_ojosIzq)
			{
				oj.push_back({ rectOjos.x + rectOjos.width / 2,  rectOjos.y + ho + rectOjos.height / 2, rectOjos.width, rectOjos.height});
				//	circle(imagen, Point(rectOjos.x + rectOjos.width / 2, rectOjos.y+ h + rectOjos.height / 2), (sqrt((rectOjos.width / 2)*(rectOjos.width / 2) + (rectOjos.height / 2) ^ 2)), Scalar(0, 255, 0), 2);
			}
			for (Rect rectOjos : vect_ojosDer)
			{
				oj.push_back({ rectOjos.x + rectOjos.width / 2,  rectOjos.y + ho + rectOjos.height / 2, rectOjos.width, rectOjos.height});
				//	circle(imagen, Point(rectOjos.x + rectOjos.width / 2, rectOjos.y+ h + rectOjos.height / 2), (sqrt((rectOjos.width / 2)*(rectOjos.width / 2) + (rectOjos.height / 2) ^ 2)), Scalar(100, 0, 0),1);
			}
			for (Rect rectOjos : vect_ojosE)
			{
				oj.push_back({ rectOjos.x + rectOjos.width / 2,  rectOjos.y + ho + rectOjos.height / 2, rectOjos.width, rectOjos.height});
				//	rectangle(imagen, Point(rectOjos.x, rectOjos.y + h), Point(rectOjos.x + rectOjos.width, (rectOjos.y+ h) + rectOjos.height), Scalar(0, 255, 0), 3);
			}
			for (Rect rectOjos : vect_ojosIzqE)
			{
				oj.push_back({ rectOjos.x + rectOjos.width / 2,  rectOjos.y + ho + rectOjos.height / 2, rectOjos.width, rectOjos.height});
				//	rectangle(imagen, Point(rectOjos.x, rectOjos.y+ h), Point(rectOjos.x + rectOjos.width, rectOjos.y + h+ rectOjos.height), Scalar(0, 100, 0), 2);
			}
			for (Rect rectOjos : vect_ojosDerE)
			{
				oj.push_back({ rectOjos.x + rectOjos.width / 2,  rectOjos.y + ho + rectOjos.height / 2, rectOjos.width, rectOjos.height});
				//	rectangle(imagen, Point(rectOjos.x, rectOjos.y+ h), Point(rectOjos.x + rectOjos.width, rectOjos.y +h+ rectOjos.height), Scalar(0, 0, 255), 1);
			}

			if (oj.empty()) {}
			else
			{
				cout << "Ojos: la size original es. " << oj.size() << endl;
				for (int j = 0; j < oj.size() - 1; j++)
				{
					for (int i = j + 1; i < oj.size(); i++)
					{
						if (oj[i].x< (oj[j].x + 12) && oj[i].x >(oj[j].x - 12))
						{
							oj.erase(oj.begin() + i);
							i--;
						}
					}
				}
			}

			for (int i = 0; i < oj.size(); i++)
			{
				rectangle(Rim_imagen, Point(oj[i].x - (oj[i].ancho / 2), oj[i].y - (oj[i].alto / 2)), Point(oj[i].x + (oj[i].ancho / 2), oj[i].y + (oj[i].alto / 2)), Scalar(0, 0, 255), 3);
				circle(Rim_imagen, Point(oj[i].x, oj[i].y), 10, Scalar(255, 0, 0), 2);
			}
		}

		if (detectBoca) {
			vector<Rect>  vect_boca, vect_boca1, vect_bocaE, vect_boca1E, vect_bocaG, vect_boca1G;										//Vector para guardar las caracteristicas de la boca
			im_boca_gris = Rim_gris(Rect(0, Rim_gris.size().height * 21 / 32, Rim_gris.size().width, Rim_gris.size().height * 5 / 16));	namedWindow("B1", WINDOW_AUTOSIZE);	moveWindow("B1", 500, 500); imshow("B1", im_boca_gris);
			//im_boca_equa = Rim_equa(Rect(0, Rim_equa.size().height * 21 / 32, Rim_equa.size().width, Rim_equa.size().height * 5 / 16));	namedWindow("B2", WINDOW_AUTOSIZE);	moveWindow("B2", 500, 500); imshow("B2", im_boca_equa);
			int hb = Rim_gris.size().height * 21 / 32;
			
			detector_boca.detectMultiScale(im_boca_gris, vect_bocaG, 1.2, 3, 0, Size(60, 60));
			//detector_boca.detectMultiScale(im_boca_equa, vect_bocaE, 1.2, 3, 0, Size(60, 60));
			//detector_boca1.detectMultiScale(im_boca_gris, vect_boca1G, 1.2, 3, 0, Size(60, 60));
			//detector_boca1.detectMultiScale(im_boca_equa, vect_boca1E, 1.2, 3, 0, Size(60, 60));
	
			for (Rect rectBoca : vect_bocaG)
			{
				bo.push_back({ rectBoca.x + rectBoca.width / 2,  rectBoca.y + hb + rectBoca.height / 2, rectBoca.width, rectBoca.height});
				ellipse(Rim_imagen, Point(rectBoca.x + rectBoca.width / 2, rectBoca.y + hb + rectBoca.height / 2), Size(rectBoca.width / 2, rectBoca.height / 2), 0, 0, 360, Scalar(255, 0, 0), 1);
			}
		}
		if (detectNariz) {
			vector<Rect> vect_nariz, vect_nariz1, vect_nariz2, vect_narizE, vect_nariz1E, vect_nariz2E;			//Vector para guardar las caracter�sticas de la nariz
			im_gris_nariz = Rim_gris(Rect(Rim_gris.size().width / 8, Rim_gris.size().height * 1 / 4, Rim_gris.size().width * 6 / 8, Rim_gris.size().height * 4 / 8));
			//Mat im_equa_nariz = Rim_equa(Rect(Rim_equa.size().width / 8, Rim_equa.size().height * 1 / 4, Rim_equa.size().width * 6 / 8, Rim_equa.size().height * 4 / 8));
			int hn = Rim_gris.size().height * 1 / 4;
			int wn = Rim_gris.size().width / 8;
			if (im_gris_nariz.size().width < 250) {
				cout << "NO DEBERIA " << endl << endl;
			}

			namedWindow("Nariz", WINDOW_AUTOSIZE);	moveWindow("Nariz", 0, 200); imshow("Nariz", im_gris_nariz);

			//detector_nariz.detectMultiScale(im_gris_nariz, vect_nariz, 1.2, 3, 0, Size(60, 60));
			//detector_nariz2.detectMultiScale(im_gris_nariz, vect_nariz1, 1.2, 3, 0, Size(60, 60));
			detector_nariz3.detectMultiScale(im_gris_nariz, vect_nariz2, 1.2, 3, 0, Size(60, 60));
			//detector_nariz.detectMultiScale(im_equa_nariz, vect_narizE, 1.2, 3, 0, Size(60, 60));
			//detector_nariz2.detectMultiScale(im_equa_nariz, vect_nariz1E, 1.2, 3, 0, Size(60, 60));
			//detector_nariz3.detectMultiScale(im_equa_nariz, vect_nariz2E, 1.2, 3, 0, Size(60, 60));

			for (Rect rectNariz : vect_nariz2)
			{
				circle(Rim_imagen, Point(rectNariz.x + wn + rectNariz.width / 2, rectNariz.y + hn + rectNariz.height / 2), (sqrt((rectNariz.width / 2)*(rectNariz.width / 2) + (rectNariz.height / 2) ^ 2)), Scalar(0, 255, 0), 1);
				circle(Rim_imagen, Point(rectNariz.x + wn + rectNariz.width / 2, rectNariz.y + hn + rectNariz.height / 2), 10, Scalar(255, 0, 0), 2);
				na.push_back({ rectNariz.x + wn + rectNariz.width / 2,  rectNariz.y + hn + rectNariz.height / 2, rectNariz.width, rectNariz.height});
				cout << "Nariz: Punto medio nariz: " << (rectNariz.x + wn + rectNariz.width / 2) << " y : " << (rectNariz.y + hn + (rectNariz.height / 2)) << " Anchura : " << rectNariz.width << " Altura : " << rectNariz.height << endl;
			}
		}



		/**************************************************************/
		/******************* REGLAS PARA IDENTIFCAR POSICION **********/
		/**************************************************************/
		if (Reglas) {
			cara Micara;

			/***************Numero de nariz********************/
			int naFlag = na.size();
			if (naFlag == 1) Micara.Cnariz = na[0];

			/***************Numero de bocas********************/
			int boFlag = bo.size();
			if (boFlag == 1)	Micara.Cboca = bo[0];

			/******Si hay m�s de una boca, identificar la correcta*********/
			if (boFlag > 1) {
				if (naFlag == 1) {
					for (int j = 1; j < bo.size(); j++)
					{
						if (abs(bo[j].x - na[0].x) < abs(bo[j - 1].x - na[0].x)) { bo.erase(bo.begin() + j - 1); }
						else { bo.erase(bo.begin() + j); }
						j--;
					}
				}
			} boFlag = bo.size();

			/*************Pintar las bocas resultantes*****/
			for (int i = 0; i < bo.size(); i++)
			{
				cout << "Reglas-Boca: Punto medio boca: " << bo[i].x << " y : " << bo[i].y << " Anchura : " << bo[i].ancho << " Altura : " << bo[i].alto << endl;
				rectangle(Rim_imagen, Point(bo[i].x - (bo[i].ancho / 2), bo[i].y - (bo[i].alto / 2)), Point(bo[i].x + (bo[i].ancho / 2), bo[i].y + (bo[i].alto / 2)), Scalar(0, 255, 0), 1);
				circle(Rim_imagen, Point(bo[i].x, bo[i].y), 10, Scalar(255, 0, 0), 3);
			}

			/************Determinar perfil***********************/
			if (naFlag == 1) {
				if (na[0].x + 12 > Rim_imagen.size().width * 3 / 5) {
					perfil = 2;
					putText(Rim_imagen, "Perfil der " + std::to_string(perfil), Point(40, 40), FONT_HERSHEY_PLAIN, 3, Scalar(0, 0, 255), 6);
				}
				else if (na[0].x - 12 < Rim_imagen.size().width * 2 / 5) {
					perfil = 1;
					putText(Rim_imagen, "Perfil izq " + std::to_string(perfil), Point(40, 40), FONT_HERSHEY_PLAIN, 3, Scalar(0, 0, 255), 6);
				}
			}
			/*	else if (boFlag == 1) {
			if (bo[0].x + 12 > imagen.size().width * 3 / 5) {
			cout << "--- Perfil der" << endl;
			perfil = 2;
			putText(imagen, "Perfil der " + std::to_string(perfil), Point(40, 40), FONT_HERSHEY_PLAIN, 3, Scalar(0, 0, 255), 6);
			}
			else if (bo[0].x - 12 < imagen.size().width * 2 / 5) {
			cout << "--- Perfil izq" << endl;
			perfil = 1;
			putText(imagen, "Perfil izq " + std::to_string(perfil), Point(40, 40), FONT_HERSHEY_PLAIN, 3, Scalar(0, 0, 255), 6);
			}
			}*/

			/****************************Reglas para los ojos***********************************/
			int ojosDer = 0; int ojosIzq = 0; int ojFlag = 0;
			if (!oj.empty())
			{
				/*****Borro los que no pueden ser ojos por geometr�a facial*****/
				for (int i = 0; i < oj.size(); i++)
				{
					if ((oj[i].x + 10 > Rim_imagen.size().width * 4 / 5) || (oj[i].x - 10 < Rim_imagen.size().width * 1 / 5))
					{
						oj.erase(oj.begin() + i);	//Borro aquellos muy alejados
						i--;
					}
					else if (perfil == 0 && naFlag == 1)
					{
						if (oj[i].x > (Rim_imagen.size().width * 2 / 5) && oj[i].x <(Rim_imagen.size().width * 3 / 5))
						{
							oj.erase(oj.begin() + i);	//Borro aquellos que coinciden con las coordenadas de la nariz (frontal)
							i--;
						}
					}
				}

				/**********Clasifico los restantes*********/
				for (int i = 0; i < oj.size(); i++)
				{
					if (naFlag == 1) {
						if (oj[i].x < na[0].x) {
							ojosIzq++;
							ojI.push_back({ oj[i].x,  oj[i].y , oj[i].ancho, oj[i].alto });
						}
						else if (oj[i].x > na[0].x) {
							ojosDer++;
							ojD.push_back({ oj[i].x,  oj[i].y , oj[i].ancho, oj[i].alto });
						}
					}
					else if (oj[i].x < Rim_imagen.size().width * 5 / 10) {
						ojosIzq++;
						ojI.push_back({ oj[i].x,  oj[i].y , oj[i].ancho, oj[i].alto });
					}
					else if (oj[i].x > Rim_imagen.size().width * 5 / 10) {
						ojosDer++;
						ojD.push_back({ oj[i].x,  oj[i].y , oj[i].ancho, oj[i].alto });
					}
				}
				//cout << "Reglas-Clasificacion Ojos: Tengo drx: " << ojosDer << " e Izq: " << ojosIzq << endl;

				/**********Eliminar ojos no correctos***************/
				if (ojosIzq > 1 && ojosDer > 1) {
					if (naFlag == 1) {
						int centroDx = 0, centroDy = 0;
						int dist_derX = 0; int dist_derY = 0; int distD = 0;
						for (int od = 0; od < ojosDer; od++)
						{
							dist_derX = dist_derX + ojD[od].x;
							dist_derY = dist_derY + ojD[od].y;
						}
						centroDx = dist_derX / ojosDer;
						centroDy = dist_derY / ojosDer;
						for (int od = 0; od < ojosDer; od++)
						{
							if (distD < sqrt(pow(abs(ojD[od].x - centroDx), 2) + (pow(abs(ojD[od].y - centroDy), 2))))
								distD = sqrt(pow(abs(ojD[od].x - centroDx), 2) + (pow(abs(ojD[od].y - centroDy), 2)));
						}
						int dist_izqX = 0; int dist_izqY = 0; int distI = 0;
						for (int oi = 0; oi < ojI.size(); oi++)
						{
							dist_izqX = dist_izqX + ojI[oi].x;
							dist_izqY = dist_izqY + ojI[oi].y;
						}
						dist_izqX = dist_izqX / ojI.size();
						dist_izqY = dist_izqY / ojI.size();
						for (int oi = 0; oi < ojI.size(); oi++)
						{
							if (distI < sqrt(pow(abs(ojI[oi].x - dist_izqX), 2) + (pow(abs(ojI[oi].y - dist_izqY), 2))))
								distI = sqrt(pow(abs(ojI[oi].x - dist_izqX), 2) + (pow(abs(ojI[oi].y - dist_izqY), 2)));
						}
						if (distI > distD) {
							int distanciaDn = pow(abs(centroDy - na[0].y), 2) + pow(abs(centroDx - na[0].x), 2);
							int distanciaI0n = pow(abs(ojI[0].y - na[0].y), 2) + pow(abs(ojI[0].x - na[0].x), 2);
							int DiferenciaDist = abs(distanciaDn - distanciaI0n);
							for (int j = 1; j < ojI.size(); j++) {
								int distanciaIjn = pow(abs(ojI[j].y - na[0].y), 2) + pow(abs(ojI[j].x - na[0].x), 2);
								if (abs(distanciaIjn - distanciaDn) >= DiferenciaDist) {
									ojI.erase(ojI.begin() + j);
									j--;
								}
								else
								{
									ojI.erase(ojI.begin());
									DiferenciaDist = abs(distanciaIjn - distanciaDn);
									j--;
								}
								ojosIzq--;
							}
						}
						if (distI < distD) {
							int distanciaIn = pow(abs(dist_izqY - na[0].y), 2) + pow(abs(dist_izqX - na[0].x), 2);
							int distanciaD0n = pow(abs(ojD[0].y - na[0].y), 2) + pow(abs(ojD[0].x - na[0].x), 2);
							int DiferenciaDist = abs(distanciaIn - distanciaD0n);
							for (int j = 1; j < ojD.size(); j++) {
								int distanciaDjn = pow(abs(ojD[j].y - na[0].y), 2) + pow(abs(ojD[j].x - na[0].x), 2);
								if (abs(distanciaDjn - distanciaIn) > DiferenciaDist) {
									ojD.erase(ojD.begin() + j);
									j--;
								}
								else
								{
									ojD.erase(ojD.begin());
									DiferenciaDist = abs(distanciaDjn - distanciaIn);
									j--;
								}
								ojosDer--;
							}
						}
					}
				}

				if (ojosIzq > 1)
				{
					if (naFlag == 1 && ojosDer == 1 && perfil == 0) {
						int distanciaDn = pow(abs(ojD[0].y - na[0].y), 2) + pow(abs(ojD[0].x - na[0].x), 2);
						int distanciaI0n = pow(abs(ojI[0].y - na[0].y), 2) + pow(abs(ojI[0].x - na[0].x), 2);
						int DiferenciaDist = abs(distanciaDn - distanciaI0n);
						for (int j = 1; j < ojI.size(); j++) {
							int distanciaIjn = pow(abs(ojI[j].y - na[0].y), 2) + pow(abs(ojI[j].x - na[0].x), 2);
							if (abs(distanciaIjn - distanciaDn) >= DiferenciaDist) {
								ojI.erase(ojI.begin() + j);
							}
							else
							{
								ojI.erase(ojI.begin());
								DiferenciaDist = abs(distanciaIjn - distanciaDn);
							}j--;
						}
					}
					else if (ojosDer == 1) {
						for (int i = 1; i < ojI.size(); i++) {
							if (abs(ojI[i].y - ojD[0].y) < abs(ojI[0].y - ojD[0].y)) {
								ojI[0] = ojI[i];
							}
							ojI.erase(ojI.begin() + i);
							i--;	
						}
					}
				}ojosIzq = ojI.size();

				if (ojosDer > 1)
				{
					if (ojosDer > 1)
					{
						if (naFlag == 1 && ojosIzq == 1 && perfil == 0) {
							int distanciaIn = pow(abs(ojI[0].y - na[0].y), 2) + pow(abs(ojI[0].x - na[0].x), 2);
							int distanciaD0n = pow(abs(ojD[0].y - na[0].y), 2) + pow(abs(ojD[0].x - na[0].x), 2);
							int DiferenciaDist = abs(distanciaIn - distanciaD0n);
							for (int j = 1; j < ojD.size(); j++) {
								int distanciaDjn = pow(abs(ojD[j].y - na[0].y), 2) + pow(abs(ojD[j].x - na[0].x), 2);
								if (abs(distanciaDjn - distanciaIn) > DiferenciaDist) {
									ojD.erase(ojD.begin() + j);
								}
								else
								{
									ojD.erase(ojD.begin());
									DiferenciaDist = abs(distanciaDjn - distanciaIn);
								}
								j--;
							}
						}
						else if (ojosIzq == 1) {
							cout << "--NO DEBERIA-- Reglas: Ojos : comparo ojos Izqs con uno derecho en y " << endl;
							for (int i = 1; i < ojD.size(); i++) {
								if (abs(ojD[i].y - ojI[0].y) < abs(ojD[0].y - ojI[0].y)) {
									ojD[0] = ojD[i];
								}
								ojD.erase(ojD.begin() + i);
								i--;
							}
						}
					}
				}ojosDer = ojD.size();
			

				/****Identificar perfiles******/
				if (ojosIzq == 1 && ojosDer == 1 && perfil == 0) {
					if (naFlag == 1) {
						//cout << "Reglas : voy a identificar perfiles con nariz" << endl;
						if (((ojI[0].y < ojD[0].y + 40) && (ojI[0].y > ojD[0].y - 40)) || ((ojD[0].y < ojI[0].y + 40) && (ojD[0].y > ojI[0].y - 40))) {
							int distanciaI = sqrt(pow(abs(ojI[0].x - na[0].x), 2) + pow(abs(ojI[0].y - na[0].y), 2));
							int distanciaD = sqrt(pow(abs(ojD[0].x - na[0].x), 2) + pow(abs(ojD[0].y - na[0].y), 2));
							if (distanciaD >(distanciaI*1.2f)) {// ((distanciaI < imagen.size().width / 10) || distanciaD >(distanciaI*1.5f)) {
								perfil = 1;
								putText(Rim_imagen, "Perfil izq", Point(40, 40), FONT_HERSHEY_PLAIN, 3, Scalar(0, 255, 0), 6);
							}
							if (distanciaI >(distanciaD*1.2f)) {//((distanciaD < imagen.size().width / 10) || distanciaI >(distanciaD*1.5f)) {
								perfil = 2;
								putText(Rim_imagen, "Perfil der ", Point(40, 40), FONT_HERSHEY_PLAIN, 3, Scalar(0, 255, 0), 6);
							}
						}
					}
					else
					{
						if (((ojI[0].y < ojD[0].y + 40) && (ojI[0].y > ojD[0].y - 40)) || ((ojD[0].y < ojI[0].y + 40) && (ojD[0].y > ojI[0].y - 40))) {
							int distanciaI, distanciaD;
							if (boFlag == 1) {
								//cout << "Reglas : voy a identificar perfiles con boca" << endl;
								distanciaI = abs(ojI[0].x - bo[0].x);
								distanciaD = abs(ojD[0].x - bo[0].x);
							}
							else {
								//cout << "Reglas : voy a identificar perfiles sin nada" << endl;
								distanciaI = abs(ojI[0].x - Rim_imagen.size().width * 5 / 10);
								distanciaD = abs(ojD[0].x - Rim_imagen.size().width * 5 / 10);
							}
							if ((distanciaI < Rim_imagen.size().width / 10) || distanciaD >(distanciaI*1.7f)) {
								perfil = 1;
								putText(Rim_imagen, "Perfil izq " , Point(40, 40), FONT_HERSHEY_PLAIN, 3, Scalar(0, 255, 0), 5);
							}
							if ((distanciaD < Rim_imagen.size().width / 10) || distanciaI >(distanciaD*1.7f)) {
								perfil = 2;
								putText(Rim_imagen, "Perfil der ", Point(40, 40), FONT_HERSHEY_PLAIN, 3, Scalar(0, 255, 0), 5);

							}
						}
					}
				}
				else if (ojosIzq == 1 && perfil == 0) {
					if (naFlag == 1) {
						int distanciaI = abs(ojI[0].x - na[0].x);
						if (distanciaI <(Rim_imagen.size().width * 1 / 20)) {
							perfil = 1;
							putText(Rim_imagen, "Perfil izq " , Point(40, 40), FONT_HERSHEY_PLAIN, 3, Scalar(0, 255, 0), 4);
						}
					}
					else
					{int distanciaI = abs(ojI[0].x - Rim_imagen.size().width * 5 / 10);
						if (distanciaI <(Rim_imagen.size().width * 1 / 20)) {
							perfil = 1;
							putText(Rim_imagen, "Perfil izq ", Point(40, 40), FONT_HERSHEY_PLAIN, 3, Scalar(0, 255, 0), 4);
						}
					}
				}
				else if (ojosDer == 1 && perfil == 0) {
					if (naFlag == 1) {
						int distanciaD = abs(ojD[0].x - na[0].x);
						if (distanciaD <(Rim_imagen.size().width * 1 / 10)) {	
							perfil = 2;
							putText(Rim_imagen, "Perfil der ", Point(40, 40), FONT_HERSHEY_PLAIN, 3, Scalar(255, 0, 0), 4);
						}
					}
					else
					{
						int distanciaD = abs(ojD[0].x - Rim_imagen.size().width * 5 / 10);
						if (distanciaD <(Rim_imagen.size().width * 1 / 20)) {
							perfil = 2;
							putText(Rim_imagen, "Perfil der ", Point(40, 40), FONT_HERSHEY_PLAIN, 3, Scalar(255, 0, 0), 4);
						}
					}
				}
				if (ojosDer == 1 && ojosIzq == 1) {
					float anguloOjos = atan(float(-ojD[0].y + ojI[0].y) / float(ojD[0].x - ojI[0].x))*180/PI;
					dist_ojos = sqrt(pow((ojI[0].y - ojD[0].y), 2) + pow((ojI[0].x - ojD[0].x), 2));
					cout << "El angulo de los ojos : " << anguloOjos << endl;
					if (alfa != -70) alfa = alfa + anguloOjos;
					else alfa = anguloOjos;
					cout << "El rostro est� girado un �ngulo : " << alfa << " tiene perfil : " << perfil << " y la distancia entre ojos " << dist_ojos / factor << endl;

				}

				/******Dibujar ojos******************/
				for (int i = 0; i < ojI.size(); i++)
				{
					cout << "Reglas: Datos Ojos x: " << ojI[i].x << " y : " << ojI[i].y << endl;
					rectangle(Rim_imagen, Point(ojI[i].x - (ojI[i].ancho / 2), ojI[i].y - (ojI[i].alto / 2)), Point(ojI[i].x + (ojI[i].ancho / 2), ojI[i].y + (ojI[i].alto / 2)), Scalar(0, 255, 0), 1);
				}
				for (int i = 0; i < ojD.size(); i++)
				{
					cout << "Reglas: Datos Ojos x: " << ojD[i].x << " y : " << ojD[i].y << endl;
					rectangle(Rim_imagen, Point(ojD[i].x - (ojD[i].ancho / 2), ojD[i].y - (ojD[i].alto / 2)), Point(ojD[i].x + (ojD[i].ancho / 2), ojD[i].y + (ojD[i].alto / 2)), Scalar(0, 255, 0), 1);
				}
			}
		}
		line(Rim_imagen, Point(Rim_imagen.size().width / 5, 0), Point(Rim_imagen.size().width / 5, Rim_imagen.size().height), Scalar(255, 0, 0), 1);
		line(Rim_imagen, Point(Rim_imagen.size().width * 2 / 5, 0), Point(Rim_imagen.size().width * 2 / 5, Rim_imagen.size().height), Scalar(255, 0, 0), 1);
		line(Rim_imagen, Point(Rim_imagen.size().width * 3 / 5, 0), Point(Rim_imagen.size().width * 3 / 5, Rim_imagen.size().height), Scalar(255, 0, 0), 1);
		line(Rim_imagen, Point(Rim_imagen.size().width * 4 / 5, 0), Point(Rim_imagen.size().width * 4 / 5, Rim_imagen.size().height), Scalar(255, 0, 0), 1);
		namedWindow("Rostro " + std::to_string(ind_read), WINDOW_AUTOSIZE);	moveWindow("Rostro " + std::to_string(ind_read), 900, 0); imshow("Rostro " + std::to_string(ind_read), Rim_imagen);



		cout << "El rostro est� girado un �ngulo : " << alfa << " tiene perfil : " << perfil << " y la distancia entre ojos " << dist_ojos / factor << endl;
		if (guardarIma)
		{
			indG++;
			index[indG] = "1212_" + std::to_string(indG) + ".png";
			try {
				imwrite(index[indG], Rim_imagen);// , compression_params);
			}
			catch (runtime_error& ex) {
				fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
				return 1;
			}
		}

		//system("PAUSE"); //Espera entrada de usuario por teclado
		int k = waitKey(0);
		if (k == 81 || k == 113)	//	q o Q para retroceder en la imagen a analizar
		{
			if (ind_read == 1) ind_read--;
			else ind_read = ind_read - 2;
		}
		else if (k == 88 || k == 120) break;		//Al presionar la tecla "x" � "X" cerramos la c�mara y salimos del programa	
													//else if (waitKey(0) == 88 || waitKey(0) == 120) break;		//Al presionar la tecla "x" � "X" cerramos la c�mara y salimos del programa	
		destroyAllWindows();
	}
}