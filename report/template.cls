\LoadClass[a4paper,12pt]{report} % Classe du document

%--------------------- Packages ------------------------
\usepackage[T5]{fontenc}      % Hỗ trợ mã hóa tiếng Việt
\usepackage[vietnamese]{babel} % Gói ngôn ngữ tiếng Việt
\usepackage{amsmath}
\RequirePackage[utf8]{inputenc} %Caractères spéciaux
\RequirePackage[section]{placeins}%Pour placement de section
\RequirePackage{mathtools} %Paquet pour des équations et symboles mathématiques
\RequirePackage{siunitx} %Pour écrire avec la notation scientifique (Ex.: \num{2e+9})
\RequirePackage{float} %Pour placement d'images
\RequirePackage{graphicx} %Paquet pour insérer des images
\RequirePackage[dvipsnames]{xcolor} % Màu sắc
\RequirePackage[justification=centering]{caption} %Pour les légendes centralisées
\RequirePackage{subcaption}
\RequirePackage{wallpaper}
\RequirePackage{nomencl}
%\makenomenclature
\RequirePackage{fancyhdr}
%\pagestyle{fancy}
%\fancyheadoffset{1cm}
%\setlength{\headheight}{2cm}
\RequirePackage{url}
\RequirePackage[hidelinks]{hyperref}%Paquet pour insérer légendes dans des sous-figures comme Figure 1a, 1b
\RequirePackage[left=2.5cm,right=2.5cm,top=2cm,bottom=3.5cm]{geometry} %Configuration de la page


%-------------------- Informations sur le rapport ----------------------

\newcommand{\UE}[1]{\renewcommand{\UE}{#1}}
\newcommand{\sujet}[1]{\renewcommand{\sujet}{#1}}
\newcommand{\titre}[1]{\renewcommand{\titre}{#1}}
\newcommand{\enseignant}[1]{\renewcommand{\enseignant}{#1}}
\newcommand{\eleves}[1]{\renewcommand{\eleves}{#1}}

\newcommand{\fairemarges}{
\makenomenclature
\pagestyle{fancy}
\fancyheadoffset{1cm}
\setlength{\headheight}{2cm}
\lhead{
        \includegraphics[scale=0.1]{./logos/logo.png}
} %Affichage de l'image au top de la page
\rhead{\nouppercase{\leftmark}}
\rfoot{\thepage}
\cfoot{\textbf{CS331} - \textbf{\titre}}
\lfoot{\UE}
\fancypagestyle{plain}{
    \fancyhf{}
    \fancyheadoffset{1cm}
    \lhead{\includegraphics[scale=0.08]{./logos/logo.png}}
    \rhead{\nouppercase{\leftmark}}
    \rfoot{\thepage}
    \cfoot{\textbf{CS331} - \textbf{\titre}}
    \lfoot{\UE}
}
}

\newcommand{\fairepagedegarde}{
\begin{titlepage}
\enlargethispage{1cm}

\ThisLRCornerWallPaper{1}{./logos/background.png}
	\centering %Centraliser le contenu
	\vspace*{-3cm}
	{\fontsize{13pt}{1}\selectfont \textbf{ĐẠI HỌC QUỐC GIA THÀNH PHỐ HỒ CHÍ MINH}\\\textbf{TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN}}\\[2cm]
	\includegraphics[scale=0.4]{./logos/uit.png}\par\vspace{1cm} 
	{\fontsize{21}{18}\selectfont 
 \textbf{BÁO CÁO ĐỒ ÁN CUỐI KỲ}
 
 \par}

	\vspace{1.5cm}
	{\scshape\Large \UE \\ \sujet \par} 
	\vspace{1cm}
    \rule{\linewidth}{0.2 mm} \\[0.4 cm]
	{\LARGE\bfseries\color{Blue} \titre \par}
    \rule{\linewidth}{0.2 mm} \\[1.5 cm]
	\vspace{-0.5cm}
    
	\begin{minipage}{0.5\textwidth}
		\begin{flushleft} \large
		\emph{\textbf{Giảng viên :}} \\
		 \enseignant \\[1cm]
		\emph{\textbf{Sinh Viên :}}\\ 
        \eleves\\ 
		\end{flushleft}
	\end{minipage}\\[1cm]
    
	\vfill
	{\large \today\par}

\end{titlepage}
}

\newcommand{\tabledematieres}{
\tableofcontents
\newpage
}

\newcommand{\insererfigure}[4]{
\begin{figure}[ht]
\centering
\includegraphics[height=#2]{#1}
\caption{#3}
\label{fig: #4}
\end{figure}
}



