export const metadata = { title: "Шлиф · классификация руд", description: "Обработка и доработка шлифов" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ru">
      <body>{children}</body>
    </html>
  );
}
